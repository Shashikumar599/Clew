"""Ingest the first training set: round-2 clean runs of the Nova campaign with the full workload.

    python3 ingest/ingest_train.py 100 20     # 100 training runs + 20 held-out happy runs

Run lists are written to data/logs/splits/ so train and held-out stay fixed.
"""
import csv, random, sys

from ingest_run import DATA, DATASET, main

FULL_WORKLOAD_COMMANDS = 36  # a complete workload run executes 36 "[START]" commands


def full_clean_round2(sub="Nova"):
    with open(DATASET / f"{sub.lower()}.csv") as f:
        for row in csv.DictReader(f):
            if row["Round_2 Failure"] != "no":
                continue
            wl = (DATASET / sub / row["Test"] / "logs" / "round_2" / "foreground_wl").glob("*.out.log*")
            n = sum("[START]" in line for p in wl for line in p.read_text(errors="replace").splitlines())
            if n == FULL_WORKLOAD_COMMANDS:
                yield row["Test"]


if __name__ == "__main__":
    n_train, n_held = int(sys.argv[1]), int(sys.argv[2])
    tests = list(full_clean_round2())
    random.Random(0).shuffle(tests)  # fixed seed: same split every time
    splits = {"nova_train": tests[:n_train], "nova_heldout": tests[n_train:n_train + n_held]}
    (DATA / "logs" / "splits").mkdir(parents=True, exist_ok=True)
    for name, ts in splits.items():
        (DATA / "logs" / "splits" / f"{name}.txt").write_text("\n".join(f"Nova-{t}-r2" for t in ts) + "\n")
        for t in ts:
            main("Nova", t, 2)
    print(f"{len(tests)} eligible Nova round-2 clean runs; ingested {n_train} train + {n_held} held-out")
