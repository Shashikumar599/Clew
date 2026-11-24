"""Run the LogProcessor over a list of runs and report plain counts.

    .venv/bin/python processing/process_runs.py data/logs/splits/nova_train.txt [max_runs]

Output: data/processed/<run_id>.jsonl (ts, service, chunk_id, template) + a console summary.
"""
import json, sys
from pathlib import Path

from log_processor import load_run, process_run

DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    run_ids = Path(sys.argv[1]).read_text().split()[: int(sys.argv[2]) if len(sys.argv) > 2 else None]
    out_dir = DATA / "processed"
    out_dir.mkdir(exist_ok=True)
    seen, total_lines, total_degenerate = set(), 0, 0
    print(f"{'run':<22}{'lines':>7}{'templates':>10}{'new':>5}{'degenerate':>11}")
    for i, rid in enumerate(run_ids):
        records = load_run(next((DATA / "logs").glob(f"*/{rid}.jsonl")))
        entries, degenerate = process_run(records)
        ids = {e["chunk_id"] for e in entries}
        with open(out_dir / f"{rid}.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in entries)
        new = len(ids - seen)  # templates not seen in any earlier run
        seen |= ids
        total_lines += len(entries)
        total_degenerate += degenerate
        if i < 5 or (i + 1) % 10 == 0:
            print(f"{rid:<22}{len(entries):>7}{len(ids):>10}{new:>5}{degenerate:>11}")
    print(f"\n{len(run_ids)} runs: {total_lines} lines, {len(seen)} distinct templates, "
          f"{total_degenerate} degenerate-template lines")
