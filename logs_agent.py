"""logs_agent (M7): RCA shortlist for one failed run (one case).

    .venv/bin/python logs_agent.py <run_id> [--repeat]

  --repeat  process a case that was already processed (simulates the same failure alerting again)

Flow (find what does not match the healthy graph):
  1. the run is already templated by the LogProcessor against the shared store (processing/process_runs.py --store)
  2. Phases 0-1, read-only, per stream (Phase 2 skipped, decision 46)
  3. matched error nodes: matched nodes with severity error
  4. evidence: failed_run_count += 1 on every node matched in this case (decision 48)
  5. budget: per-case input-token ceiling + daily budget for the flow type; over budget -> the span
     stays unmatched, never truncated
  6. Phase 3: every unmatched span (decision 45) -> failure-tuned prompt (severity error = origin,
     warn = symptom, info = harmless)
  7. each Phase 3 step becomes a node with creator logs_agent, linked to the steps around it
     (so Phase 0 can find the step next time)
  8. shortlist -> data/cases/<run_id>.json
"""
import argparse, json, math, shutil, sys
from datetime import date

import kg_config as cfg
import kg_builder as kb
from kg.graph import Graph
from kg.node import new_case_node, now_ms
from matcher.llm import BedrockLLM, CachedLLM
from matcher.segmenter import segment_run
from storage.node_repo import NodeRepo


def load_budget():
    """Tokens logs_agent spent today for this flow type (global budget, reset daily)."""
    today = date.today().isoformat()
    st = json.loads(cfg.BUDGET_PATH.read_text()) if cfg.BUDGET_PATH.exists() else {}
    return st if st.get("day") == today else {"day": today, "input_tokens": 0}


def save_budget(st):
    cfg.BUDGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg.BUDGET_PATH.write_text(json.dumps(st))


def estimate_tokens(ents, template):
    """Rough input size of a span before calling (~3.5 characters per token), to check the budget first."""
    windows = math.ceil(len(ents) / cfg.WINDOW_LINES)
    chars = sum(len(e["template"]) + len(e["service"]) + 10 for e in ents) + windows * len(template)
    return int(chars / 3.5)


def run_case(run_id, repeat=False):
    entries = kb.load_processed(run_id)
    repo = NodeRepo(cfg.GRAPH_DIR, cfg.GRAPH_ID)
    graph = Graph(repo, cfg.FLOW_TYPE)
    root = repo.get(graph.root_id)
    cases = root.setdefault("cases", [])
    if run_id in cases and not repeat:
        sys.exit(f"{run_id} was already processed (use --repeat to process it again)")

    # Phases 0-1, read-only (the same walk kg_builder uses).
    walks = kb.walk_streams(entries, repo, graph)
    funnel = kb.funnel_of(walks)

    # Matched error nodes + evidence counters (decision 48).
    matched, error_nodes = set(), {}
    for s, w in walks.items():
        for v in w["visits"]:
            n = repo.get(v["node"])
            matched.add(v["node"])
            if n["severity"] == "error":
                e = error_nodes.setdefault(v["node"], {
                    "node": v["node"], "service": s, "title": n["title"], "summary": n["summary"],
                    "fix_suggestion": n["fix_suggestion"], "creator": n["creator"],
                    "healthy_runs": n["run_count"], "failed_runs_before": n.get("failed_run_count", 0),
                    "first_ts": w["ents"][v["lines"][0]]["ts"], "visits": 0, "lines": 0})
                e["visits"] += 1
                e["lines"] += len(v["lines"])
    for nid in matched:
        n = repo.get(nid)
        n["failed_run_count"] = n.get("failed_run_count", 0) + 1
        n["last_failed_case"] = run_id
    cases.append(run_id)

    # Unmatched spans in time order (earliest first: closest to the origin if the budget runs out).
    tasks = []
    for s, w in walks.items():
        by_end = {v["lines"][-1]: v for v in w["visits"]}
        by_start = {v["lines"][0]: v for v in w["visits"]}
        for a, b in w["spans"]:
            tasks.append((w["ents"][a]["ts"], s, a, b, by_end.get(a - 1), by_start.get(b + 1)))
    tasks.sort(key=lambda t: t[0])

    budget, spent = load_budget(), 0
    llm = CachedLLM(BedrockLLM(cfg.CASE_MODEL_ID, cfg.AWS_PROFILE, cfg.AWS_REGION, cfg.MAX_OUTPUT_TOKENS,
                               cfg.TEMPERATURE), cfg.LLM_CACHE_DIR, cfg.CASE_MODEL_ID)
    template = cfg.CASE_PROMPT_PATH.read_text()
    steps_out, left = [], []
    tin = tout = new_nodes = 0
    title = lambda v: repo.get(v["node"])["title"] if v else None
    for ts, s, a, b, before, after in tasks:
        ents = walks[s]["ents"][a:b + 1]
        span = {"service": s, "first_ts": ents[0]["ts"], "last_ts": ents[-1]["ts"], "lines": len(ents)}
        est = estimate_tokens(ents, template)
        # Budget: checked before Phase 3; over budget -> skip, never truncate.
        if spent + est > cfg.CASE_TOKEN_CEILING or budget["input_tokens"] + est > cfg.DAILY_TOKEN_BUDGET:
            left.append({**span, "reason": "budget"})
            continue
        calls_before = llm.calls
        try:
            steps, st = segment_run(ents, llm, template, cfg.WINDOW_LINES, cfg.MAX_ATTEMPTS, log=lambda *x: None,
                                    before_title=title(before), after_title=title(after), service=s)
        except RuntimeError as e:  # invalid reply after the retry: the span stays unmatched (decision 39)
            steps, st = [], {**getattr(e, "stats", {"input_tokens": 0, "output_tokens": 0}), "invalid": True}
            left.append({**span, "reason": "invalid reply"})
        if llm.calls > calls_before:  # cached replies cost nothing
            spent += st["input_tokens"]
            budget["input_tokens"] += st["input_tokens"]
            tin += st["input_tokens"]
            tout += st["output_tokens"]

        # Write the steps as logs_agent nodes and link them to the steps around the span.
        prev = before["node"] if before else graph.service_root(s)
        for step in steps:
            node = new_case_node(repo.graph_id, step, len(repo.all()), run_id, s)
            existing = repo.get(node["chunk_id"])
            if existing is None:
                repo.put(node)
                new_nodes += 1
                matched.add(node["chunk_id"])  # counted once for this case
            elif node["chunk_id"] not in matched:  # same step already in the graph: count the case once
                existing["failed_run_count"] = existing.get("failed_run_count", 0) + 1
                existing["last_failed_case"] = run_id
                matched.add(node["chunk_id"])
            graph.add_transition(prev, node["chunk_id"])
            prev = node["chunk_id"]
            n = repo.get(node["chunk_id"])
            steps_out.append({"service": s, "first_ts": step["lines"][0]["ts"], "last_ts": step["lines"][-1]["ts"],
                              "lines": len(step["lines"]), "severity": step["severity"], "title": step["title"],
                              "summary": step["summary"], "fix_suggestion": step["fix_suggestion"],
                              "node": node["chunk_id"], "new_node": existing is None,
                              "healthy_runs": n["run_count"], "failed_runs": n.get("failed_run_count", 0)})
        if steps and after:
            graph.add_transition(prev, after["node"])

    # Shortlist: matched error nodes + what Phase 3 explained + what is still unmatched.
    rank = {"error": 0, "warn": 1, "info": 2}
    steps_out.sort(key=lambda x: (rank[x["severity"]], x["first_ts"]))
    shortlist = {"run_id": run_id, "graph_id": cfg.GRAPH_ID, "repeat": run_id in cases[:-1],
                 "lines": len(entries), "funnel": funnel, "spans": len(tasks),
                 "matched_error_nodes": sorted(error_nodes.values(), key=lambda e: e["first_ts"]),
                 "phase3_steps": steps_out, "still_unmatched": left,
                 "llm": {"model": cfg.CASE_MODEL_ID, "calls": llm.calls, "cached": llm.hits,
                         "input_tokens": tin, "output_tokens": tout, "cost_usd": round(kb.cost_of(tin, tout), 3)},
                 "nodes_written": new_nodes, "nodes_matched": len(matched)}
    cfg.CASES_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.CASES_DIR / f"{run_id}{'.repeat' + str(cases.count(run_id) - 1) if shortlist['repeat'] else ''}.json"
    out.write_text(json.dumps(shortlist, indent=1))
    backup = repo.path.with_name(f"{cfg.GRAPH_ID}.before-case-{run_id}.json")
    if repo.path.exists():
        shutil.copy(repo.path, backup)
    repo.flush()
    save_budget(budget)
    report(shortlist, out)


def report(sl, out):
    f = sl["funnel"]
    p0 = f["phase0_exact"] + f["phase0_regex"] + f["phase0_fuzzy"]
    print(f"case {sl['run_id']}{' (repeat)' if sl['repeat'] else ''}: {sl['lines']} lines | Phase 0 {p0}, "
          f"Phase 1 {f['phase1']}, unmatched {f['unmatched']} in {sl['spans']} spans")
    print(f"\nMATCHED ERROR NODES ({len(sl['matched_error_nodes'])})")
    for e in sl["matched_error_nodes"]:
        print(f"  {e['first_ts'][11:23]} {e['service']:<24} {e['title']}  [{e['creator']}; seen in "
              f"{e['healthy_runs']} healthy / {e['failed_runs_before'] + 1} failed runs]")
    print(f"\nPHASE 3 STEPS ({len(sl['phase3_steps'])}; error = origin, warn = symptom)")
    for x in sl["phase3_steps"]:
        print(f"  {x['severity']:<5} {x['first_ts'][11:23]} {x['service']:<24} {x['title']} ({x['lines']} lines)"
              f"{'' if x['new_node'] else '  [known node]'}")
        if x["severity"] != "info":
            print(f"        {x['summary']}")
            if x["fix_suggestion"]:
                print(f"        fix: {x['fix_suggestion']}")
    if sl["still_unmatched"]:
        print(f"\nSTILL UNMATCHED ({len(sl['still_unmatched'])} spans)")
        for x in sl["still_unmatched"]:
            print(f"  {x['first_ts'][11:23]} {x['service']:<24} {x['lines']} lines ({x['reason']})")
    L = sl["llm"]
    print(f"\nLLM: {L['calls']} calls ({L['cached']} cached), {L['input_tokens']} in / {L['output_tokens']} out "
          f"(~${L['cost_usd']}); nodes written {sl['nodes_written']}, nodes matched {sl['nodes_matched']}")
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_id")
    p.add_argument("--repeat", action="store_true")
    a = p.parse_args()
    run_case(a.run_id, a.repeat)
