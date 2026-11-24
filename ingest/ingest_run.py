"""Ingest one run (one test round) into the local log store.

Stand-in for "logs in Elasticsearch": every service log of the run is merged by timestamp
into one JSONL file, placed in happy/ or alert/ by the run's own outcome.

    python3 ingest/ingest_run.py Nova Test_1 1
"""
import csv, json, re, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
DATASET = DATA / "Fault-Injection-Dataset"
OUT = DATA / "logs"

# Every log line record starts with "YYYY-MM-DD HH:MM:SS.ffff"; anything else is a continuation.
TS = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
# Logs are written in local time, and the machine's zone differs per campaign. Measured from the
# HTTP "Date: ... GMT" headers inside the logs (41 sampled runs, constant within each subsystem).
TZ_OFFSET_HOURS = {"Nova": 2, "Cinder": -4, "Neutron": -4}


def to_utc(raw, tz):
    # pad/trim fraction to microseconds so "03:21:07.558" and "03:21:38.044576" both parse
    date, frac = raw.split(".")
    dt = datetime.strptime(f"{date}.{frac[:6].ljust(6, '0')}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.replace(tzinfo=tz).astimezone(timezone.utc).isoformat(timespec="microseconds")


def service_name(path, round_dir):
    # nova/nova-api.log.bzip2.out -> nova/nova-api ; foreground_wl/..out.log.. -> workload/out
    rel = path.relative_to(round_dir)
    if rel.parts[0] == "foreground_wl":
        kind = "err" if ".err.log" in rel.name else "out"
        return f"workload/{kind}"
    return f"{rel.parts[0]}/{rel.name.split('.log')[0]}"


def read_records(path, service, tz):
    """Yield (ts_utc, service, file, message). A line without a timestamp belongs to the
    closest timestamped line above it in the same file."""
    rec = None
    for line in path.read_text(errors="replace").splitlines():
        m = TS.match(line)
        if m:
            if rec:
                yield rec
            rec = [to_utc(m.group(1), tz), service, path.name, line]
        elif rec:
            rec[3] += "\n" + line
        # lines before the first timestamp (none seen so far) are dropped
    if rec:
        yield rec


def run_failed(subsystem, test, rnd):
    with open(DATASET / f"{subsystem.lower()}.csv") as f:
        for row in csv.DictReader(f):
            if row["Test"] == test:
                return row[f"Round_{rnd} Failure"] == "yes"
    raise SystemExit(f"{test} not found in {subsystem.lower()}.csv")


def main(subsystem, test, rnd):
    round_dir = DATASET / subsystem / test / "logs" / f"round_{rnd}"
    tz = timezone(timedelta(hours=TZ_OFFSET_HOURS[subsystem]))
    records = []
    for path in sorted(round_dir.glob("*/*.out")):
        records.extend(read_records(path, service_name(path, round_dir), tz))
    # stable sort by time; ties keep file order then line order, so the result is deterministic
    records.sort(key=lambda r: r[0])

    run_id = f"{subsystem}-{test}-r{rnd}"
    folder = "alert" if run_failed(subsystem, test, rnd) else "happy"
    out = OUT / folder / f"{run_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ts, service, file, message in records:
            f.write(json.dumps({"ts": ts, "service": service, "file": file, "message": message}) + "\n")
    print(f"{run_id}: {len(records)} records -> {out.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
