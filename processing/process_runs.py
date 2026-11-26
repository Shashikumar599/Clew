"""Run the LogProcessor over a list of runs and report plain counts.

    .venv/bin/python processing/process_runs.py <run_list> [max_runs] [--store]

Without --store: fresh Drain3 miner per run (decision 16) -> data/processed/<run_id>.jsonl
With --store:    template against the shared store (decision 36) -> data/processed_store/<run_id>.jsonl
Each line: ts, service, chunk_id (sha256 of the line template), template.
"""
import json, sys
from pathlib import Path

import log_processor_config as cfg
from log_processor import load_run, load_store, process_run

DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_store = "--store" in sys.argv
    run_ids = Path(args[0]).read_text().split()[: int(args[1]) if len(args) > 1 else None]
    store = load_store(cfg.DRAIN_STORE_PATH) if use_store else None
    out_dir = DATA / ("processed_store" if use_store else "processed")
    out_dir.mkdir(exist_ok=True)
    seen, total_lines, total_degenerate, total_unseen, total_boiler = set(), 0, 0, 0, 0
    print(f"{'run':<22}{'lines':>7}{'templates':>10}{'new':>5}{'degenerate':>11}{'unseen':>8}{'dropped':>8}")
    for i, rid in enumerate(run_ids):
        records = load_run(next((DATA / "logs").glob(f"*/{rid}.jsonl")))
        entries, stats = process_run(records, store)
        ids = {e["chunk_id"] for e in entries}
        with open(out_dir / f"{rid}.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)
        new = len(ids - seen)  # templates not seen in any earlier run
        seen |= ids
        total_lines += len(entries)
        total_degenerate += stats["degenerate"]
        total_unseen += stats["unseen"]
        total_boiler += stats["boilerplate"]
        if i < 5 or (i + 1) % 10 == 0 or stats["boilerplate"]:
            print(f"{rid:<22}{len(entries):>7}{len(ids):>10}{new:>5}{stats['degenerate']:>11}{stats['unseen']:>8}{stats['boilerplate']:>8}")
    print(f"\n{len(run_ids)} runs: {total_lines} lines, {len(seen)} distinct templates, "
          f"{total_degenerate} degenerate-template lines, {total_unseen} lines no store cluster fits, "
          f"{total_boiler} boilerplate lines dropped -> {out_dir}")
