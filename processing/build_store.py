"""Build the shared Drain3 template store from a batch of runs (decision 36).

    .venv/bin/python processing/build_store.py

Learns every line of every run listed in DRAIN_STORE_RUNS (the 100 training runs), in list order,
and saves the store to DRAIN_STORE_PATH. Held-out runs are not used.
"""
from pathlib import Path

import log_processor_config as cfg
from log_processor import learn_store, load_run, new_miner, save_store

DATA = Path(__file__).resolve().parent.parent / "data"

if __name__ == "__main__":
    run_ids = cfg.DRAIN_STORE_RUNS.read_text().split()
    miner = new_miner()
    for i, rid in enumerate(run_ids, 1):
        learn_store(miner, load_run(next((DATA / "logs").glob(f"*/{rid}.jsonl"))))
        if i % 10 == 0:
            print(f"{i:>3} runs learned, {len(miner.drain.clusters)} clusters")
    save_store(miner, cfg.DRAIN_STORE_PATH)
    print(f"saved {cfg.DRAIN_STORE_PATH} ({len(miner.drain.clusters)} clusters from {len(run_ids)} runs)")
