"""LogProcessor (M3): reduce one run's merged logs to an ordered list of template entries.

Every input comes from log_processor_config.py. Steps, in the doc's order:
extract -> group by pod -> strip nulls -> (collapse API noise: left out) -> boilerplate filter
-> burst-window ordering -> Drain3 templating with masking -> degenerate-template guard.
"""
import hashlib, json, re
from datetime import datetime
from pathlib import Path

from drain3 import TemplateMiner
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
    """Filter known-boilerplate prefixes (no-op while the configured list is empty)."""
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


def new_miner():
    """A fresh Drain3 state (per run, not persisted; the graph nodes are what persists)."""
    config = TemplateMinerConfig()
    config.profiling_enabled = False
    config.drain_depth = cfg.DRAIN_DEPTH
    config.drain_sim_th = cfg.DRAIN_SIM_THRESHOLD
    config.drain_max_children = cfg.DRAIN_MAX_CHILDREN
    config.drain_max_clusters = cfg.DRAIN_MAX_CLUSTERS
    config.masking_instructions = [MaskingInstruction(rx, name) for name, rx in cfg.MASKS]
    return TemplateMiner(None, config)


def literal_tokens(template):
    return [t for t in template.split() if not re.fullmatch(r"<[A-Z0-9_*]+>", t)]


def process_run(records):
    """Template every line of one run, in burst-window order. Returns (entries, degenerate_count)."""
    records = burst_merge(drop_boilerplate(strip_nulls(records)), cfg.BURST_WINDOW_SECONDS)
    miner = new_miner()
    entries, degenerate = [], 0
    for r in records:
        content = content_of(r)
        template = miner.add_log_message(content)["template_mined"]  # per-line templating
        if len(literal_tokens(template)) < cfg.MIN_LITERAL_TOKENS:
            # too generic: it would match almost anything, so keep the (masked) raw message
            template = miner.masker.mask(content)
            degenerate += 1
        entries.append({"ts": r["ts"], "service": r[cfg.GROUP_BY_FIELD],
                        "chunk_id": hashlib.sha256(template.encode()).hexdigest(), "template": template})
    return entries, degenerate
