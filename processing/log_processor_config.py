"""LogProcessor inputs, one section per step, in the order they run.

Everything is left empty on purpose: fill in each value as the step is decided.
A value of None / [] means "step not configured yet".
Input record (from ingestion): {"ts", "service", "file", "message"}.
"""

# 1. Extract -- which fields of an ingested record the processor reads.
EXTRACT_FIELDS = ["ts", "service", "message"]  # `file` is not needed: `service` already names the pod

# 2. Group by pod -- the field that identifies a pod (a service log file in this project).
GROUP_BY_FIELD = "service"

# 3. Strip null fields -- remove keys with empty values before processing.
STRIP_NULL_FIELDS = False  # our records never have empty fields

# 4. Collapse repeated API-response noise.
COLLAPSE_API_NOISE = None  # left out (decided)

# 5. Filter known-boilerplate prefixes (health checks, environment dumps).
#    Prefix list still empty; the CONTAINS list below drops config dumps (decision 18, revisited).
#    A line is dropped when its content starts with one of these.
BOILERPLATE_PREFIXES = []  # e.g. ["DEBUG some.module "]
#    Which text the prefixes are matched against.
BOILERPLATE_MATCH_ON = None  # e.g. "content" (after timestamp/pid/request id) or "raw"
#    Lines whose raw first line CONTAINS one of these are dropped too (decision 18, revisited).
#    "log_opt_values" = the config library's startup dump of every option (an environment dump):
#    9 of 124 Nova runs, 770 lines per dump in placement-api (up to 10,640 per run with keystone).
BOILERPLATE_CONTAINS = ["log_opt_values"]

# 6. Drain3 templating.
#    Values below are reasonable starting points (Drain3's own defaults where it has one).
DRAIN_INPUT = "level module message"  # LEVEL + module + message; timestamp, pid and request context left out
DRAIN_MULTILINE = "first_line"  # feed the first line of a multi-line record only
DRAIN_DEPTH = 4
DRAIN_SIM_THRESHOLD = 0.4
DRAIN_MAX_CHILDREN = 100
DRAIN_MAX_CLUSTERS = None  # unlimited
DRAIN_LEARNING = "shared store"  # decision 36: one store learned from a batch, saved, every run templated against it
from pathlib import Path as _Path
DRAIN_STORE_PATH = _Path(__file__).resolve().parent.parent / "data" / "drain3" / "nova-workload-v1.bin"
DRAIN_STORE_RUNS = _Path(__file__).resolve().parent.parent / "data" / "logs" / "splits" / "nova_train.txt"

# 7. Masking: variable instance data (ids, addresses, timestamps) becomes placeholders.
#    (name, regex) pairs; each match is replaced by <name>.
#    Order matters: specific patterns first. A bare "mask every number" rule is deliberately left out.
MASKS = [
    ("UUID", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    ("HEX32", r"\b[0-9a-fA-F]{32}\b"),  # UUID without dashes (project / user ids)
    ("DATETIME", r"\b\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?Z?\b"),  # timestamps inside a message
    ("IP", r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),  # needs a dot in the domain, so host@name is not caught
    ("EPOCH", r"\b1[0-9]{9}\b"),  # 10-digit epoch used as suffix of generated resource names
]

# 8. Degenerate-template guard: fewer than 3 literal tokens (wildcards stripped)
#    means the template is too generic and the raw message is kept instead.
MIN_LITERAL_TOKENS = 3

# 9. Ordering: burst-window merge.
BURST_WINDOW_SECONDS = 1.0  # starting point; gap between consecutive lines of the current pod
