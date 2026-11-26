"""kg_builder (M4): build and grow the knowledge graph from happy runs.

    .venv/bin/python kg_builder.py create <run_id> [--max-windows N] [--show-prompt]
    .venv/bin/python kg_builder.py match <run_id>     (Phase 0 + 1 only, read-only, no LLM)
    .venv/bin/python kg_builder.py add <run_id>       (Phase 0 + 1, LLM for unmatched spans, graph update)

  --show-prompt    print the first window's prompt and exit (no LLM call)
  --max-windows N  per stream: stop after N windows and do NOT save the graph (a cheap test)

Stream mode (cfg.STREAM_BY = "service", decision 38): each service log file is its own stream, in
time order. Segmentation, the walk, spans and edges all stay inside one stream; the graph is
ROOT -> one sub-root per stream -> that stream's steps. Steps of different streams that overlap in
time are counted on the nodes (overlap_counts). With STREAM_BY = None the whole run is one stream
from the root (how graphs v1 and v2 were built).
"""
import argparse, json, shutil, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import kg_config as cfg
from kg.graph import Graph
from kg.node import absorb_line, partial_shape, prefix_cut, widen_counts
from matcher.llm import BedrockLLM, CachedLLM
from matcher.segmenter import build_prompt, segment_run
from matcher.walk import GraphIndex, match_run
from storage.node_repo import NodeRepo

WORKERS = 2  # parallel LLM calls; 4 hit Bedrock throttling


def load_processed(run_id):
    path = cfg.PROCESSED_DIR / f"{run_id}.jsonl"
    if not path.exists():
        sys.exit(f"no processed run at {path} (run processing/process_runs.py first)")
    return [json.loads(line) for line in path.read_text().splitlines()]


def streams_of(entries):
    """{stream: [entry + its position in the run]}; one stream (None) when STREAM_BY is None."""
    out = defaultdict(list)
    for i, e in enumerate(entries):
        out[e[cfg.STREAM_BY] if cfg.STREAM_BY else None].append({**e, "idx": i})
    return dict(sorted(out.items(), key=lambda x: str(x[0])))


def stream_root(graph, s, create=True):
    return graph.root_id if s is None else graph.service_root(s, create)


def new_llm():
    return CachedLLM(BedrockLLM(cfg.MODEL_ID, cfg.AWS_PROFILE, cfg.AWS_REGION, cfg.MAX_OUTPUT_TOKENS,
                                cfg.TEMPERATURE), cfg.LLM_CACHE_DIR, cfg.MODEL_ID)


def cost_of(tin, tout):
    return (tin * cfg.PRICE_IN_PER_M + tout * cfg.PRICE_OUT_PER_M) / 1e6  # list price, report only


def spans_of(lines):
    """Consecutive unmatched lines -> (start, end) inclusive positions in `lines`."""
    spans, start = [], None
    for k, r in enumerate(lines + [{"phase": "end"}]):
        if r["phase"] == "unmatched":
            start = k if start is None else start
        elif start is not None:
            spans.append((start, k - 1))
            start = None
    return spans


def span_of(lines):
    return lines[0]["ts"], lines[-1]["ts"]


def create(run_id, max_windows=None, show_prompt=False):
    entries = load_processed(run_id)
    streams = streams_of(entries)
    template = cfg.PROMPT_PATH.read_text()
    if show_prompt:
        s, ents = next(iter(streams.items()))
        print(build_prompt(template, ents[:cfg.WINDOW_LINES], len(ents) <= cfg.WINDOW_LINES, None, None, s))
        return
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    if graph.real_nodes():
        sys.exit(f"{repo.path} already has nodes: 'create' builds an empty graph only (use 'add')")

    print(f"run {run_id}: {len(entries)} lines in {len(streams)} streams -> graph {cfg.GRAPH_ID} ({cfg.MODEL_ID})")
    llm = new_llm()

    def seg(item):
        s, ents = item
        steps, st = segment_run(ents, llm, template, cfg.WINDOW_LINES, cfg.MAX_ATTEMPTS, max_windows,
                                log=lambda *a: None, service=s)
        print(f"  {str(s):<32} {len(ents):>6} lines {st['windows']:>3} windows {len(steps):>4} steps "
              f"{st['retries']} retries", flush=True)
        return s, steps, st

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(seg, streams.items()))

    # Stages 5 + 6: per stream, walk its steps in order from its sub-root.
    created = revisits = 0
    items = []
    for s, steps, _ in results:
        prev = stream_root(graph, s)
        for step in steps:
            nid, is_new = graph.visit(step, run_id, s)
            created += is_new
            revisits += not is_new
            graph.add_transition(prev, nid)
            prev = nid
            items.append((s, nid, *span_of(step["lines"])))
    pairs = graph.record_overlaps(items) if cfg.STREAM_BY else 0
    root = repo.get(graph.root_id)
    root["run_count"] += 1
    root["runs"] = [run_id]

    tin = sum(st["input_tokens"] for _, _, st in results)
    tout = sum(st["output_tokens"] for _, _, st in results)
    nodes = graph.real_nodes()
    edges = sum(len(n["child_chunk_ids"]) for n in repo.all())
    print(f"\nLLM: {sum(st['windows'] for *_, st in results)} windows, {llm.calls} calls, {llm.hits} cached, "
          f"{sum(st['retries'] for *_, st in results)} retries, {tin} input / {tout} output tokens (~${cost_of(tin, tout):.2f})")
    print(f"steps: {sum(len(x[1]) for x in results)} covering {sum(len(st['lines']) for x in results for st in x[1])} "
          f"of {len(entries)} lines")
    print(f"nodes: {created} created, {revisits} revisits | graph: {len(nodes)} nodes + root + {len(streams) if cfg.STREAM_BY else 0} "
          f"sub-roots, {edges} edges, {sum(n['is_loop'] for n in nodes)} with is_loop, {pairs} overlapping step pairs")
    sev = defaultdict(int)
    for n in nodes:
        sev[n["severity"]] += 1
    print(f"severity: {dict(sev)}")
    if max_windows is not None:
        print("\n--max-windows: partial run, graph NOT saved")
        return
    repo.flush()
    print(f"\nsaved {repo.path}")


def walk_streams(entries, repo, graph):
    """Phase 0 + 1 per stream. Returns {stream: {ents, lines, visits, spans}}; positions are per stream."""
    out = {}
    for s, ents in streams_of(entries).items():
        sub = stream_root(graph, s, create=False)
        if sub is None:  # a stream the graph has never seen: every line unmatched
            lines, visits = [{"i": k, "phase": "unmatched"} for k in range(len(ents))], []
        else:
            idx = GraphIndex(repo, sub, cfg, service=s if cfg.STREAM_BY else None)
            lines, visits, _ = match_run(ents, idx)
        out[s] = {"ents": ents, "lines": lines, "visits": visits, "spans": spans_of(lines)}
    return out


def funnel_of(walks):
    lines = [r for w in walks.values() for r in w["lines"]]
    c = lambda pred: sum(1 for r in lines if pred(r))
    return {"phase0_exact": c(lambda r: r["phase"] == "phase0" and r.get("method") == "exact"),
            "phase0_regex": c(lambda r: r["phase"] == "phase0" and r.get("method") == "regex"),
            "phase0_fuzzy": c(lambda r: r["phase"] == "phase0" and r.get("method") == "fuzzy"),
            "phase1": c(lambda r: r["phase"] == "phase1"), "absorbed": c(lambda r: r["phase"] == "absorbed"),
            "unmatched": c(lambda r: r["phase"] == "unmatched")}


def match(run_id):
    """Walk a run through the graph with Phase 0 + Phase 1 only. Read-only: no LLM, no graph writes."""
    entries = load_processed(run_id)
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    if not graph.real_nodes():
        sys.exit(f"{repo.path} has no nodes: run 'create' first")
    walks = walk_streams(entries, repo, graph)
    f, n = funnel_of(walks), len(entries)
    p0 = f["phase0_exact"] + f["phase0_regex"] + f["phase0_fuzzy"]
    pct = lambda k: f"{k:>6} ({100 * k / n:5.1f}%)"
    sizes = [e - s + 1 for w in walks.values() for s, e in w["spans"]]
    bucket = lambda lo, hi: sum(1 for x in sizes if lo <= x <= hi)
    print(f"run {run_id}: {n} lines in {len(walks)} streams vs graph {cfg.GRAPH_ID} ({len(graph.real_nodes())} nodes)\n")
    print("funnel")
    for k in ("phase0_exact", "phase0_regex", "phase0_fuzzy", "phase1", "unmatched"):
        print(f"  {k:<14} {pct(f[k])}")
    print(f"\nunmatched spans: {len(sizes)} (1 line: {bucket(1, 1)}, 2-5: {bucket(2, 5)}, 6-20: {bucket(6, 20)}, "
          f"21+: {bucket(21, 10**9)}); largest {sorted(sizes)[-5:][::-1]}")
    partial = new_edges = 0
    nodes = {x["chunk_id"]: x for x in repo.all()}
    for s, w in walks.items():
        prev = stream_root(graph, s, create=False)
        for v in w["visits"]:
            node = nodes[v["node"]]
            partial += len(v["counts"]) < len(node["line_templates"])
            new_edges += prev is not None and v["node"] not in nodes[prev]["child_chunk_ids"]
            prev = v["node"]
    print(f"visits: {sum(len(w['visits']) for w in walks.values())}, partial {partial}; "
          f"transitions not yet in the graph: {new_edges} (counting across unmatched spans)")
    print(f"\n{'stream':<32}{'lines':>7}{'phase0':>8}{'phase1':>8}{'unmatched':>10}")
    for s, w in sorted(walks.items(), key=lambda x: -len(x[1]["ents"])):
        c = lambda ph: sum(1 for r in w["lines"] if r["phase"] == ph)
        print(f"{str(s):<32}{len(w['ents']):>7}{c('phase0'):>8}{c('phase1'):>8}{c('unmatched'):>10}")
    cfg.MATCH_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.MATCH_DIR / f"{run_id}.json"
    out.write_text(json.dumps({"run_id": run_id, "graph_id": cfg.GRAPH_ID, "lines_total": n, "funnel": f,
                               "streams": {str(s): {"lines": len(w["ents"]),
                                                    "spans": [[w["ents"][a]["idx"], w["ents"][b]["idx"]] for a, b in w["spans"]],
                                                    "lines_detail": [{**r, "i": w["ents"][r["i"]]["idx"],
                                                                      **({"node": nodes[r["node"]]["seq"]} if "node" in r else {})}
                                                                     for r in w["lines"]]}
                                           for s, w in walks.items()}}, indent=1))
    print(f"\nsaved {out}  (line indices are positions in data/{cfg.PROCESSED_DIR.name}/{run_id}.jsonl)")


def add(run_id):
    """Grow the graph with a later happy run (decision 28): walk each stream (Phase 0 + 1), send only
    unmatched spans to the LLM with the titles of the stream's steps around them (Phase 3), then
    update in stream order: matched visits widen counts; Phase 3 steps become nodes; edges follow
    the stream from its sub-root; overlapping steps of different streams are counted."""
    entries = load_processed(run_id)
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    root = repo.get(graph.root_id)
    if run_id in root.get("runs", []):
        sys.exit(f"{run_id} is already in the graph (runs: {root['runs']})")
    walks = walk_streams(entries, repo, graph)
    edges_before = sum(len(n["child_chunk_ids"]) for n in repo.all())
    heal = Counter()
    if cfg.SELF_HEAL:
        # Between the cheap phases and Phase 3, single-line spans are tested for absorption into the
        # preceding node as an optional line, capped (decision 40). Absorbed lines skip the LLM.
        for s, w in walks.items():
            by_end = {v["lines"][-1]: v for v in w["visits"]}
            keep = []
            for a, b in w["spans"]:
                v = by_end.get(a - 1)
                if a == b and v is not None:
                    tpl = w["ents"][a]["template"]
                    r = absorb_line(repo.get(v["node"]), tpl, cfg.OPTIONAL_CAP)
                    heal["absorb_" + r] += 1
                    if r != "refused":
                        v["lines"].append(a)
                        v["counts"][tpl] += 1
                        w["lines"][a] = {"i": a, "phase": "absorbed", "node": v["node"]}
                        continue
                keep.append((a, b))
            w["spans"] = keep
    f = funnel_of(walks)
    nodes_before = {n["chunk_id"]: n for n in repo.all()}
    title = lambda v: nodes_before[v["node"]]["title"] if v else None

    tasks = []
    for s, w in walks.items():
        by_end = {v["lines"][-1]: v for v in w["visits"]}
        by_start = {v["lines"][0]: v for v in w["visits"]}
        for a, b in w["spans"]:
            tasks.append((s, a, b, title(by_end.get(a - 1)), title(by_start.get(b + 1))))
    print(f"run {run_id}: {len(entries)} lines, {len(walks)} streams; Phase 0+1 left {f['unmatched']} lines "
          f"in {len(tasks)} spans -> LLM ({WORKERS} in parallel)")

    llm, template = new_llm(), cfg.PROMPT_PATH.read_text()

    def segment(t):
        s, a, b, before, after = t
        try:
            return segment_run(walks[s]["ents"][a:b + 1], llm, template, cfg.WINDOW_LINES, cfg.MAX_ATTEMPTS,
                               log=lambda *x: None, before_title=before, after_title=after, service=s)
        except RuntimeError as e:
            # Decision 39: an invalid reply after the retry leaves this span unmatched (like an
            # over-budget skip) and the run goes on; the span is counted in the build log.
            print(f"  {s} lines {walks[s]['ents'][a]['idx']}-{walks[s]['ents'][b]['idx']}: left unmatched ({e})")
            return [], {**getattr(e, "stats", {"input_tokens": 0, "output_tokens": 0}), "invalid": True}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(segment, tasks))
    invalid = sum(1 for _, st in results if st.get("invalid"))
    tin = sum(st["input_tokens"] for _, st in results)
    tout = sum(st["output_tokens"] for _, st in results)

    # split_node (decision 40): a visit where only a prefix of the node's required templates
    # matched splits the node into matched + residual. One split per node per run (the first one).
    shapes, alias = Counter(), {}
    if cfg.SELF_HEAL:
        for s, w in walks.items():
            for v in w["visits"]:
                node = repo.get(v["node"])
                shapes[partial_shape(node, v["counts"])] += 1
                cut = None if v["node"] in alias else prefix_cut(node, v["counts"])
                if cut is not None:
                    alias[v["node"]] = graph.split_node(v["node"], cut)
                    heal["splits"] += 1

    # Per stream: matched visits + Phase 3 steps at their span positions, in stream order.
    events = defaultdict(list)
    for s, w in walks.items():
        events[s] += [(v["lines"][0], "visit", v) for v in w["visits"]]
    for (s, a, *_), (steps, _) in zip(tasks, results):
        pos = a
        for step in steps:
            events[s].append((pos, "step", step))
            pos += len(step["lines"])

    widened = created = revisited = partial = 0
    items = []
    for s, evs in events.items():
        ents = walks[s]["ents"]
        prev = stream_root(graph, s)
        for _, kind, obj in sorted(evs, key=lambda x: x[0]):
            if kind == "visit":
                ts = (ents[obj["lines"][0]]["ts"], ents[obj["lines"][-1]]["ts"])
                parts = [(obj["node"], obj["counts"])]
                if obj["node"] in alias:  # the node was split this run: the visit goes to N1 (-> N2)
                    n1, n2 = alias[obj["node"]]
                    t1 = {lt["template"] for lt in repo.get(n1)["line_templates"]}
                    parts = [(x, c) for x, c in ((n1, {t: k for t, k in obj["counts"].items() if t in t1}),
                                                  (n2, {t: k for t, k in obj["counts"].items() if t not in t1})) if c]
                for nid, counts in parts:
                    node = repo.get(nid)
                    partial += len(counts) < len(node["line_templates"])
                    lo, rf = widen_counts(node, counts, sum(counts.values()), run_id, loosen=not cfg.SELF_HEAL,
                                          cap=cfg.OPTIONAL_CAP if cfg.LOOSEN_CAPPED else None)
                    heal["loosened"] += lo
                    heal["loosen_refused"] += rf
                    widened += 1
                    graph.add_transition(prev, nid)
                    prev = nid
                    items.append((s, nid, *ts))
                continue
            else:
                nid, is_new = graph.visit(obj, run_id, s)
                created += is_new
                revisited += not is_new
                ts = span_of(obj["lines"])
            graph.add_transition(prev, nid)
            prev = nid
            items.append((s, nid, *ts))
    pairs = graph.record_overlaps(items) if cfg.STREAM_BY else 0
    root["run_count"] += 1
    root.setdefault("runs", []).append(run_id)

    # Promotion (logs_agent -> kg_builder when a failure-discovered node reappears in healthy runs), gated
    # by evidence on both sides (decision 48): >= PROMOTE_MIN_HEALTHY healthy sightings and a healthy share
    # (healthy runs / training runs) at least as high as its failed share (cases / all cases).
    promoted = []
    n_healthy, n_cases = len(root["runs"]), max(1, len(root.get("cases", [])))
    for n in repo.all():
        if n.get("creator") == "logs_agent" and n.get("last_run_id") == run_id:
            if (n["run_count"] >= cfg.PROMOTE_MIN_HEALTHY and
                    n["run_count"] / n_healthy >= n.get("failed_run_count", 0) / n_cases):
                n.update({"creator": "kg_builder", "promoted_from": "logs_agent", "promoted_in_run": run_id})
                promoted.append(n["title"])

    nodes = graph.real_nodes()
    edges = sum(len(n["child_chunk_ids"]) for n in repo.all())
    n_steps = sum(len(st) for st, _ in results)
    p0 = f["phase0_exact"] + f["phase0_regex"] + f["phase0_fuzzy"]
    print(f"funnel: phase0 {p0} ({100 * p0 / len(entries):.1f}%), phase1 {f['phase1']}, unmatched {f['unmatched']}")
    print(f"Phase 3: {len(tasks)} spans -> {n_steps} steps ({invalid} spans left unmatched: invalid reply); {llm.calls} calls ({llm.hits} cached); "
          f"{tin} input / {tout} output tokens (~${cost_of(tin, tout):.2f})")
    print(f"graph update: {widened} visits widened ({partial} partial); {created} new nodes, "
          f"{revisited} Phase 3 steps matched an existing chunk_id; {pairs} overlapping step pairs")
    if cfg.SELF_HEAL:
        print(f"self-healing: {heal['absorb_absorbed']} lines absorbed as optional, {heal['absorb_known']} into a "
              f"node that had the template, {heal['absorb_refused']} refused (cap); {heal['splits']} nodes split; "
              f"missing lines made optional {heal['loosened']}, refused (cap) {heal['loosen_refused']}")
        print(f"visit shapes (required templates used): {dict(shapes)}")
    seen_case = sum(1 for n in repo.all() if n.get("creator") == "logs_agent" and n.get("last_run_id") == run_id)
    if seen_case or promoted:
        print(f"logs_agent nodes seen in this healthy run: {seen_case}; promoted: {promoted}")
    print(f"graph now: {len(nodes)} nodes, {edges} edges (+{edges - edges_before}), "
          f"{sum(n['is_loop'] for n in nodes)} with is_loop; runs: {len(root['runs'])}")

    row = {"run_id": run_id, "lines": len(entries), "phase0": p0, "phase1": f["phase1"], "unmatched": f["unmatched"],
           "spans": len(tasks), "single_line_spans": sum(1 for _, a, b, *_ in tasks if a == b),
           "phase3_steps": n_steps, "invalid_spans": invalid, "new_nodes": created, "phase3_existing": revisited, "visits": widened,
           "partial_visits": partial, "new_edges": edges - edges_before, "nodes_total": len(nodes),
           "edges_total": edges, "is_loop_total": sum(n["is_loop"] for n in nodes), "llm_calls": llm.calls,
           "absorbed": heal["absorb_absorbed"], "absorbed_known": heal["absorb_known"],
           "absorb_refused": heal["absorb_refused"], "splits": heal["splits"], "loosened": heal["loosened"], "promoted": promoted, "loosen_refused": heal["loosen_refused"], "visit_shapes": dict(shapes),
           "input_tokens": tin, "output_tokens": tout, "cost_usd": round(cost_of(tin, tout), 3)}
    cfg.MATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.MATCH_DIR / "build_log.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    backup = repo.path.with_name(f"{cfg.GRAPH_ID}.before-{run_id}.json")
    if repo.path.exists():
        shutil.copy(repo.path, backup)  # keep the previous graph to compare or roll back
    repo.flush()
    print(f"saved {repo.path} (previous version: {backup.name})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["create", "match", "add"])
    p.add_argument("run_id")
    p.add_argument("--max-windows", type=int)
    p.add_argument("--show-prompt", action="store_true")
    a = p.parse_args()
    {"create": lambda: create(a.run_id, a.max_windows, a.show_prompt),
     "match": lambda: match(a.run_id), "add": lambda: add(a.run_id)}[a.command]()
