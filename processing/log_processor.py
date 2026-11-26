"""LogProcessor (M3): reduce one run's merged logs to an ordered list of template entries.

Every input comes from log_processor_config.py. Steps, in order:
extract -> group by pod -> strip nulls -> (collapse API noise: left out) -> boilerplate filter
-> burst-window ordering -> Drain3 templating with masking -> degenerate-template guard.
"""
import hashlib, json, re
from datetime import datetime
from pathlib import Path

from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

import log_processor_config as cfg

# "<date> <time> <pid> LEVEL module [req ctx] message"  (oslo.log format, most services)
OSLO = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \d+ (\w+) (\S+) (?:\[[^\]]*\] )?(.*)$")
# "<date> <time> LEVEL [sample_workload.sh]: message"  (workload log)
WORKLOAD = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ (\w+) \[sample_workload\.sh\]: (.*)$")


def load_run(path):
    """Extract: keep only the configured fields of each ingested record."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return [{k: r[k] for k in cfg.EXTRACT_FIELDS} for r in records]


def strip_nulls(records):
    if not cfg.STRIP_NULL_FIELDS:
        return records
    return [{k: v for k, v in r.items() if v not in (None, "")} for r in records]


def content_of(record):
    """What Drain3 sees ("level module message"): first line only, without timestamp, pid, request ids."""
    first = record["message"].split("\n", 1)[0]
    m = OSLO.match(first)
    if m:
        return f"{m.group(1)} {m.group(2)} {m.group(3)}"
    m = WORKLOAD.match(first)
    return f"{m.group(1)} workload {m.group(2)}" if m else first


def drop_boilerplate(records):
    """Filter known boilerplate: lines whose raw first line contains a BOILERPLATE_CONTAINS marker
    (decision 18, revisited), then lines starting with a BOILERPLATE_PREFIXES entry (list empty)."""
    if cfg.BOILERPLATE_CONTAINS:
        marks = cfg.BOILERPLATE_CONTAINS
        records = [r for r in records if not any(m in r["message"].split("\n", 1)[0] for m in marks)]
    if not cfg.BOILERPLATE_PREFIXES:
        return records
    prefixes = tuple(cfg.BOILERPLATE_PREFIXES)
    return [r for r in records if not content_of(r).startswith(prefixes)]


def burst_merge(records, window_s):
    """Stay on the current pod while its next line is within `window_s` of the previous one;
    when it goes quiet, switch to the pod whose next line has the earliest timestamp."""
    pods = {}
    for r in records:  # ingestion output is time-sorted, so each pod's list is too
        pods.setdefault(r[cfg.GROUP_BY_FIELD], []).append((datetime.fromisoformat(r["ts"]), r))
    pos = {p: 0 for p in pods}
    out, current, last = [], None, None
    while len(out) < len(records):
        nxt = pods[current][pos[current]][0] if current and pos[current] < len(pods[current]) else None
        if nxt is not None and (nxt - last).total_seconds() <= window_s:
            pod = current  # still in its burst
        else:  # pod went quiet: switch to the pod with the next-earliest timestamp
            pod = min((p for p in pods if pos[p] < len(pods[p])), key=lambda p: pods[p][pos[p]][0])
        t, rec = pods[pod][pos[pod]]
        pos[pod] += 1
        out.append(rec)
        current, last = pod, t
    return out


def new_miner(persistence=None):
    """A Drain3 miner. With `persistence` (a FilePersistence) the saved store is loaded on creation."""
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.snapshot_interval_minutes = 10 ** 6  # the store is saved explicitly, never mid-batch
    config.drain_depth = cfg.DRAIN_DEPTH
    config.drain_sim_th = cfg.DRAIN_SIM_THRESHOLD
    config.drain_max_children = cfg.DRAIN_MAX_CHILDREN
    config.drain_max_clusters = cfg.DRAIN_MAX_CLUSTERS
    config.masking_instructions = [MaskingInstruction(rx, name) for name, rx in cfg.MASKS]
    return TemplateMiner(persistence, config)


def ordered(records):
    """Steps before templating: strip nulls -> boilerplate filter -> burst-window order."""
    return burst_merge(drop_boilerplate(strip_nulls(records)), cfg.BURST_WINDOW_SECONDS)


# Shared template store (decision 36). Drain3 is shared,
# not rebuilt per run: we learn one store from a batch of runs, save it, then template
# every run against the final store, so the same line gets the same template in every run
# (re-templating against the final state stops templates drifting within a batch).
def learn_store(miner, records):
    """Add one run's lines to the shared store."""
    for r in ordered(records):
        miner.add_log_message(content_of(r))


def save_store(miner, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    miner.persistence_handler = FilePersistence(str(path))
    miner.save_state("store built")


def load_store(path):
    if not path.exists():
        raise SystemExit(f"no template store at {path} (run processing/build_store.py first)")
    return new_miner(FilePersistence(str(path)))


def literal_tokens(template):
    return [t for t in template.split() if not re.fullmatch(r"<[A-Z0-9_*]+>", t)]


def process_run(records, store=None):
    """Template every line of one run, in burst-window order. Returns (entries, stats).
    store=None: fresh miner per run (decision 16, superseded). store given: match each line against
    the frozen store; a line no cluster fits keeps its masked raw text and is counted as unseen."""
    miner = store or new_miner()
    kept = ordered(records)
    entries, stats = [], {"degenerate": 0, "unseen": 0, "boilerplate": len(records) - len(kept)}
    for r in kept:
        content = content_of(r)
        if store is None:
            template = miner.add_log_message(content)["template_mined"]
        else:
            cluster = store.match(content, full_search_strategy="fallback")
            if cluster is None:
                template = store.masker.mask(content)
                stats["unseen"] += 1
            else:
                template = cluster.get_template()
        if len(literal_tokens(template)) < cfg.MIN_LITERAL_TOKENS:
            # too generic: it would match almost anything, so keep the (masked) raw message
            template = miner.masker.mask(content)
            stats["degenerate"] += 1
        entries.append({"ts": r["ts"], "service": r[cfg.GROUP_BY_FIELD],
                        "chunk_id": hashlib.sha256(template.encode()).hexdigest(), "template": template})
    return entries, stats
