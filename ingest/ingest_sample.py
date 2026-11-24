"""Ingest a small, reproducible sample: N alert runs and N happy runs, spread over the subsystems.

    python3 ingest/ingest_sample.py 20
"""
import csv, random, shutil, sys
from pathlib import Path

from ingest_run import DATA, DATASET, main

SUBSYSTEMS = ["Nova", "Cinder", "Neutron"]


def candidates():
    """(subsystem, test, round, failed) for every run in the dataset."""
    for sub in SUBSYSTEMS:
        with open(DATASET / f"{sub.lower()}.csv") as f:
            for row in csv.DictReader(f):
                for rnd in (1, 2):
                    yield sub, row["Test"], rnd, row[f"Round_{rnd} Failure"] == "yes"


def pick(runs, n, rng):
    # round-robin over subsystems so the sample is not all Nova
    by_sub = {s: [r for r in runs if r[0] == s] for s in SUBSYSTEMS}
    for lst in by_sub.values():
        rng.shuffle(lst)
    return [by_sub[SUBSYSTEMS[i % 3]].pop() for i in range(n)]


def copy_ground_truth(sub, test, rnd):
    # scoring only, never read by the RCA code; kept outside data/logs
    dst = DATA / "ground_truth" / f"{sub}-{test}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATASET / sub / test / "fip_info.data", dst)
    trigger = DATASET / sub / test / "logs" / "round_1" / "trigger_log"
    if trigger.exists():
        shutil.copy(trigger, dst)


if __name__ == "__main__":
    n = int(sys.argv[1])
    rng = random.Random(0)  # fixed seed: same sample every time
    runs = list(candidates())
    # alert = failed round (any round); happy = clean round. Leaked round-2 failures are alerts too.
    sample = pick([r for r in runs if r[3]], n, rng) + pick([r for r in runs if not r[3]], n, rng)
    for sub, test, rnd, _ in sample:
        main(sub, test, rnd)
        copy_ground_truth(sub, test, rnd)
