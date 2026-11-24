"""kg_builder (M4, step 1): create the knowledge graph from ONE processed happy run.

    .venv/bin/python kg_builder.py create <run_id> [--max-windows N] [--show-prompt]
    .venv/bin/python kg_builder.py match <run_id>     (Phase 0 + 1 only, read-only, no LLM)
    .venv/bin/python kg_builder.py add <run_id>       (Phase 0 + 1, LLM for unmatched spans, graph update)

  --show-prompt    print the first window's prompt and exit (no LLM call)
  --max-windows N  stop after N windows and do NOT save the graph (a cheap test; replies are cached,
                   so a later full run reuses them)

Pipeline: processed lines -> LLM segmentation in windows (stage 4) -> steps to nodes (stage 5)
-> edges in run order from the sentinel root (stage 6) -> NodeRepo file (stage 7).
Adding later runs through the matcher (Phase 0 + 1, decision 28) is the next step, not built yet.
"""
import argparse, json, shutil, sys
from concurrent.futures import ThreadPoolExecutor

import kg_config as cfg
from kg.graph import Graph
from kg.node import widen_counts
from matcher.llm import BedrockLLM, CachedLLM
from matcher.segmenter import build_prompt, segment_run
from matcher.walk import GraphIndex, match_run
from storage.node_repo import NodeRepo


def load_processed(run_id):
    path = cfg.PROCESSED_DIR / f"{run_id}.jsonl"
    if not path.exists():
        sys.exit(f"no processed run at {path} (run processing/process_runs.py first)")
    return [json.loads(line) for line in path.read_text().splitlines()]


def create(run_id, max_windows=None, show_prompt=False):
    entries = load_processed(run_id)
    template = cfg.PROMPT_PATH.read_text()
    if show_prompt:
        first = entries[:cfg.WINDOW_LINES]
        print(build_prompt(template, first, len(entries) <= cfg.WINDOW_LINES, None, None))
        return

    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    if graph.real_nodes():
        sys.exit(f"{repo.path} already has nodes: 'create' builds an empty graph only "
                 f"(adding runs comes with the matcher)")

    print(f"run {run_id}: {len(entries)} lines -> graph {cfg.GRAPH_ID}, windows of {cfg.WINDOW_LINES}")
    llm = CachedLLM(BedrockLLM(cfg.MODEL_ID, cfg.AWS_PROFILE, cfg.AWS_REGION, cfg.MAX_OUTPUT_TOKENS,
                               cfg.TEMPERATURE), cfg.LLM_CACHE_DIR, cfg.MODEL_ID)
    steps, stats = segment_run(entries, llm, template, cfg.WINDOW_LINES, cfg.MAX_ATTEMPTS, max_windows)

    # Stages 5 + 6: walk the run's steps in order from the root, creating or revisiting nodes.
    prev, created, revisits = graph.root_id, 0, 0
    for step in steps:
        chunk_id, is_new = graph.visit(step, run_id)
        created += is_new
        revisits += not is_new
        graph.add_transition(prev, chunk_id)
        prev = chunk_id
    root = repo.get(graph.root_id)
    root["run_count"] += 1
    root["runs"] = [run_id]  # which runs are in the graph (guards against adding one twice)

    nodes = graph.real_nodes()
    edges = sum(len(n["child_chunk_ids"]) for n in repo.all())
    covered = sum(len(s["lines"]) for s in steps)
    print(f"\nLLM: {stats['windows']} windows, {llm.calls} calls, {llm.hits} cached, {stats['retries']} retries, "
          f"{stats['input_tokens']} input / {stats['output_tokens']} output tokens")
    print(f"steps: {len(steps)} covering {covered} of {len(entries)} lines")
    print(f"nodes: {created} created, {revisits} revisits of an existing node")
    print(f"graph: {len(nodes)} nodes + root, {edges} edges, {sum(n['is_loop'] for n in nodes)} with is_loop")
    by_sev = {}
    for n in nodes:
        by_sev[n["severity"]] = by_sev.get(n["severity"], 0) + 1
    print(f"severity: {by_sev}")

    if max_windows is not None:
        print("\n--max-windows: partial run, graph NOT saved")
        return
    repo.flush()
    print(f"\nsaved {repo.path}")


def spans_of(lines):
    """Consecutive unmatched lines -> (start, end) inclusive line indices."""
    spans, start = [], None
    for r in lines + [{"phase": "end"}]:
        if r["phase"] == "unmatched":
            start = r["i"] if start is None else start
        elif start is not None:
            spans.append((start, prev))
            start = None
        prev = r.get("i")
    return spans


def match(run_id):
    """Walk a run through the graph with Phase 0 + Phase 1 only. Read-only: no LLM, no graph writes.
    Prints the funnel and saves data/matches/<run_id>.json for review."""
    entries = load_processed(run_id)
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    if not graph.real_nodes():
        sys.exit(f"{repo.path} has no nodes: run 'create' first")
    idx = GraphIndex(repo, graph.root_id, cfg)
    lines, visits, transitions = match_run(entries, idx)
    seq = lambda nid: idx.nodes[nid]["seq"]

    n = len(entries)
    count = lambda pred: sum(1 for r in lines if pred(r))
    p0 = {m: count(lambda r, m=m: r["phase"] == "phase0" and r.get("method") == m) for m in ("exact", "regex", "fuzzy")}
    p1, um = count(lambda r: r["phase"] == "phase1"), count(lambda r: r["phase"] == "unmatched")
    spans = spans_of(lines)
    sizes = [e - s + 1 for s, e in spans]
    bucket = lambda lo, hi: sum(1 for x in sizes if lo <= x <= hi)
    pct = lambda k: f"{k:>6} ({100 * k / n:5.1f}%)"

    print(f"run {run_id}: {n} lines vs graph {cfg.GRAPH_ID} ({len(graph.real_nodes())} nodes)\n")
    print("funnel")
    print(f"  Phase 0 exact    {pct(p0['exact'])}")
    print(f"  Phase 0 regex    {pct(p0['regex'])}")
    print(f"  Phase 0 fuzzy    {pct(p0['fuzzy'])}")
    print(f"  Phase 1 fuzzy    {pct(p1)}")
    print(f"  unmatched        {pct(um)}   -> would go to Phase 2 / absorption / Phase 3")
    print(f"\nunmatched spans: {len(spans)}  (1 line: {bucket(1, 1)}, 2-5: {bucket(2, 5)}, 6-20: {bucket(6, 20)}, "
          f"21-127: {bucket(21, 127)}, 128+: {bucket(128, 10**9)}); largest {sorted(sizes)[-5:][::-1] if sizes else []}")
    windows = -(-um // cfg.WINDOW_LINES)
    print(f"LLM estimate if spans are packed into shared windows: {windows} windows, ~${windows * cfg.WINDOW_COST_USD:.2f}")

    # What kg_builder would change in the graph (not applied).
    real_edges = [t for t in transitions if not t["after_gap"]]
    new_edges = {(t["from"], t["to"]) for t in real_edges if t["to"] not in idx.nodes[t["from"]]["child_chunk_ids"]}
    visited = {v["node"] for v in visits}
    below = above = partial = 0
    for v in visits:
        node, k = idx.nodes[v["node"]], len(v["lines"])
        below += k < node["line_budget"]["min"]
        above += k > node["line_budget"]["max"]
        partial += len(v["counts"]) < len(node["line_templates"])
    print(f"\nvisits: {len(visits)} to {len(visited)} distinct nodes; graph nodes not visited: {len(graph.real_nodes()) - len(visited)}")
    print(f"  visits with fewer lines than the node's min: {below}, more than its max: {above}; "
          f"visits that used only some of the node's templates: {partial}")
    print(f"transitions: {len(real_edges)} between matched lines ({len(new_edges)} distinct edges not yet in the graph), "
          f"{len(transitions) - len(real_edges)} across an unmatched span")

    cfg.MATCH_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.MATCH_DIR / f"{run_id}.json"
    out.write_text(json.dumps({
        "run_id": run_id, "graph_id": cfg.GRAPH_ID, "lines_total": n,
        "funnel": {"phase0_exact": p0["exact"], "phase0_regex": p0["regex"], "phase0_fuzzy": p0["fuzzy"],
                   "phase1": p1, "unmatched": um},
        "spans": [{"start": s, "end": e, "lines": e - s + 1} for s, e in spans],
        "new_edges": [[seq(a), seq(b)] for a, b in sorted(new_edges, key=lambda x: (seq(x[0]), seq(x[1])))],
        "visits": [{"node_seq": seq(v["node"]), "first": v["lines"][0], "last": v["lines"][-1], "lines": len(v["lines"])}
                   for v in visits],
        "lines": [{**r, "node": seq(r["node"])} if "node" in r else r for r in lines],
    }, indent=1))
    print(f"\nsaved {out}  (line indices are 0-based positions in data/processed/{run_id}.jsonl)")


def add(run_id, workers=4):
    """kg_builder for a later happy run (decision 28): walk it through the graph (Phase 0 + 1),
    send only the unmatched spans to the LLM (Phase 3), then update the graph in run order:
    matched visits widen their node's counts; Phase 3 steps become new nodes (or revisit one with
    the same chunk_id); edges follow the run's sequence of steps from the root, so a new node is
    wired between the step before its span and the step after it."""
    entries = load_processed(run_id)
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    root = repo.get(graph.root_id)
    if run_id in root.get("runs", []):
        sys.exit(f"{run_id} is already in the graph (runs: {root['runs']})")
    idx = GraphIndex(repo, graph.root_id, cfg)
    lines, visits, transitions = match_run(entries, idx)
    spans = spans_of(lines)
    by_end = {v["lines"][-1]: v for v in visits}
    by_start = {v["lines"][0]: v for v in visits}
    title = lambda v: idx.nodes[v["node"]]["title"] if v else None
    print(f"run {run_id}: {len(entries)} lines; Phase 0+1 left {sum(e - s + 1 for s, e in spans)} lines "
          f"in {len(spans)} spans -> LLM ({workers} in parallel)")

    # Phase 3: one segmentation call per span, with the titles of the known steps around it (doc).
    llm = CachedLLM(BedrockLLM(cfg.MODEL_ID, cfg.AWS_PROFILE, cfg.AWS_REGION, cfg.MAX_OUTPUT_TOKENS,
                               cfg.TEMPERATURE), cfg.LLM_CACHE_DIR, cfg.MODEL_ID)
    template = cfg.PROMPT_PATH.read_text()

    def segment(span):
        s, e = span
        return segment_run(entries[s:e + 1], llm, template, cfg.WINDOW_LINES, cfg.MAX_ATTEMPTS, log=lambda *a: None,
                           before_title=title(by_end.get(s - 1)), after_title=title(by_start.get(e + 1)))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(segment, spans))  # in span order; any invalid reply stops the run here
    tokens_in = sum(st["input_tokens"] for _, st in results)
    tokens_out = sum(st["output_tokens"] for _, st in results)

    # The run as one ordered sequence of steps: matched visits + Phase 3 steps at their span positions.
    events = [(v["lines"][0], "visit", v) for v in visits]
    for (s, _), (steps, _) in zip(spans, results):
        pos = s
        for step in steps:
            events.append((pos, "step", step))
            pos += len(step["lines"])
    events.sort(key=lambda x: x[0])

    widened = created = revisited = 0
    edges_before = sum(len(n["child_chunk_ids"]) for n in repo.all())
    prev = graph.root_id
    for _, kind, obj in events:
        if kind == "visit":
            nid = obj["node"]
            widen_counts(repo.get(nid), obj["counts"], len(obj["lines"]), run_id)
            widened += 1
        else:
            nid, is_new = graph.visit(obj, run_id)
            created += is_new
            revisited += not is_new
        graph.add_transition(prev, nid)
        prev = nid
    root["run_count"] += 1
    root.setdefault("runs", []).append(run_id)

    nodes = graph.real_nodes()
    edges = sum(len(n["child_chunk_ids"]) for n in repo.all())
    cost = tokens_in * 5 / 1e6 + tokens_out * 25 / 1e6  # Opus 4.6 list price, for the report only
    n_steps = sum(len(st) for st, _ in results)
    print(f"\nPhase 3: {len(spans)} spans -> {n_steps} steps; {llm.calls} calls ({llm.hits} cached); "
          f"{tokens_in} input / {tokens_out} output tokens (~${cost:.2f} at list price)")
    print(f"graph update: {widened} visits widened existing nodes; {created} new nodes, "
          f"{revisited} Phase 3 steps matched an existing chunk_id")
    print(f"graph now: {len(nodes)} nodes + root, {edges} edges (+{edges - edges_before}), "
          f"{sum(n['is_loop'] for n in nodes)} with is_loop; runs: {root['runs']}")

    # One metrics row per added run (for the convergence table).
    funnel = lambda ph, m=None: sum(1 for r in lines if r["phase"] == ph and (m is None or r.get("method") == m))
    row = {"run_id": run_id, "lines": len(entries), "phase0": funnel("phase0"), "phase1": funnel("phase1"),
           "unmatched": funnel("unmatched"), "spans": len(spans), "single_line_spans": sum(1 for s, e in spans if s == e),
           "phase3_steps": n_steps, "new_nodes": created, "phase3_existing": revisited, "visits": len(visits),
           "partial_visits": sum(1 for v in visits if len(v["counts"]) < len(idx.nodes[v["node"]]["line_templates"])),
           "new_edges": edges - edges_before, "nodes_total": len(nodes), "edges_total": edges,
           "is_loop_total": sum(n["is_loop"] for n in nodes), "llm_calls": llm.calls,
           "input_tokens": tokens_in, "output_tokens": tokens_out, "cost_usd": round(cost, 3)}
    cfg.MATCH_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.MATCH_DIR / "build_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")

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
    if a.command == "create":
        create(a.run_id, a.max_windows, a.show_prompt)
    elif a.command == "match":
        match(a.run_id)
    else:
        add(a.run_id)
