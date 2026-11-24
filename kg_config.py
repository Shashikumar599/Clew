"""Inputs for building the knowledge graph (M4, stages 4-7). One place for every value."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA = ROOT_DIR / "data"

# Graph identity -- doc: graph_id = "{flow_type}#{version}" (decision 30).
FLOW_TYPE = "nova-workload"  # one flow type = the whole workload (decision 5)
GRAPH_ID = f"{FLOW_TYPE}#v1"  # v1 = first graph, built from the Nova campaign

# Where things live (all under the gitignored data/ folder).
PROCESSED_DIR = DATA / "processed"  # LogProcessor output, one JSONL per run (M3)
GRAPH_DIR = DATA / "graph"  # one JSON file per graph (decision 26)
LLM_CACHE_DIR = DATA / "llm_cache"  # every accepted LLM reply, keyed by the prompt's hash

# LLM segmentation (decisions 20-22, 25).
PROMPT_PATH = ROOT_DIR / "matcher" / "prompts" / "segmentation.txt"
WINDOW_LINES = 128  # lines per LLM call, carry-over included
AWS_PROFILE = "dev"  # overridden by the AWS_PROFILE environment variable
AWS_REGION = "ap-south-1"
MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.0  # same window in, same steps out (as far as the model allows)
MAX_ATTEMPTS = 2  # an invalid reply is retried once with the error appended, then the build stops

# Matcher: Phase 0 + Phase 1 (decisions 24, 28, 31-34).
MATCH_DIR = DATA / "matches"  # per-run match result (line -> node), written by `kg_builder.py match`
PREFIX_TOKENS = 6  # doc: prefix-agreement score over the first six literal tokens
FUZZY_BASE = 0.8  # threshold for short lines
FUZZY_LONG_LINE = 30  # lines with more tokens than this get a lower threshold ...
FUZZY_DECAY_PER_TOKEN = 0.01  # ... by this much per extra token ...
FUZZY_FLOOR = 0.6  # ... but never below this
FUZZY_RELAX = 0.05  # lower again once half of the node's templates are confirmed in the current visit
COVERAGE_MIN = 0.5  # a visit must cover at least half of the node's distinct templates (approach A)
EXTEND_SCAN_FACTOR = 4  # look-ahead stops after factor x node's max lines + extra (cost bound only)
EXTEND_SCAN_EXTRA = 10
WINDOW_COST_USD = 0.11  # ~cost of one 128-line Opus window, measured on run 1 (for the estimate only)
