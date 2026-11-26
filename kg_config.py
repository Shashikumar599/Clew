"""Inputs for building the knowledge graph (M4, stages 4-7). One place for every value."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA = ROOT_DIR / "data"

# Graph identity: graph_id = "{flow_type}#{version}" (decision 30).
FLOW_TYPE = "nova-workload"  # one flow type = the whole workload (decision 5)
# v1 = first graph, templates from a fresh Drain3 miner per run (data/processed/), kept for comparison.
# v2 = same flow rebuilt from the shared template store (decision 36, data/processed_store/).
# v3 = shared store + one stream per service log file (decision 38).
# v4 = v3's run-1 graph, runs 2-9 re-added with self-healing: split_node + capped absorption (decision 40).
# v5 = v4 + a missing required line becomes optional under the same cap as absorption (decision 41).
GRAPH_VERSION = os.environ.get("CLEW_GRAPH_VERSION", "v3")  # decision 44: v3 is the base graph  # override to build/match another version
STREAM_BY = "service" if GRAPH_VERSION not in ("v1", "v2") else None  # None = whole run as one stream
GRAPH_ID = f"{FLOW_TYPE}#{GRAPH_VERSION}"

# Where things live (all under the gitignored data/ folder).
PROCESSED_DIR = DATA / ("processed" if GRAPH_VERSION == "v1" else "processed_store")  # LogProcessor output (M3)
GRAPH_DIR = DATA / "graph"  # one JSON file per graph (decision 26)
LLM_CACHE_DIR = DATA / "llm_cache"  # every accepted LLM reply, keyed by the prompt's hash

# LLM segmentation (decisions 20-22, 25).
PROMPT_PATH = ROOT_DIR / "matcher" / "prompts" / ("segmentation_stream.txt" if STREAM_BY else "segmentation.txt")
WINDOW_LINES = 128  # lines per LLM call, carry-over included
AWS_PROFILE = "dev"  # overridden by the AWS_PROFILE environment variable
AWS_REGION = "ap-south-1"
MODEL_ID = os.environ.get("CLEW_LLM_MODEL")  # Bedrock model id for segmentation (decision 37); set in the environment
PRICE_IN_PER_M, PRICE_OUT_PER_M = 3.0, 15.0  # list price per 1M tokens, for cost reports only
MAX_OUTPUT_TOKENS = 8192
TEMPERATURE = 0.0  # same window in, same steps out (as far as the model allows)
MAX_ATTEMPTS = 2  # an invalid reply is retried once with the error appended, then the build stops

# Matcher: Phase 0 + Phase 1 (decisions 24, 28, 31-34).
MATCH_DIR = DATA / "matches" / GRAPH_VERSION  # per-run match results + build_log.jsonl (v1 files sit in data/matches/)
PREFIX_TOKENS = 6  # prefix-agreement score over the first six literal tokens
FUZZY_BASE = 0.8  # threshold for short lines
FUZZY_LONG_LINE = 30  # lines with more tokens than this get a lower threshold ...
FUZZY_DECAY_PER_TOKEN = 0.01  # ... by this much per extra token ...
FUZZY_FLOOR = 0.6  # ... but never below this
FUZZY_RELAX = 0.05  # lower again once half of the node's templates are confirmed in the current visit
COVERAGE_MIN = 0.5  # a visit must cover at least half of the node's distinct templates (approach A)
EXTEND_SCAN_FACTOR = 4  # look-ahead stops after factor x node's max lines + extra (cost bound only)
EXTEND_SCAN_EXTRA = 10
# Self-healing in kg_builder (decision 40). v1-v3 instead loosen: a template missing from a visit gets min=0.
SELF_HEAL = GRAPH_VERSION not in ("v1", "v2", "v3")
LOOSEN_CAPPED = GRAPH_VERSION not in ("v1", "v2", "v3", "v4")  # missing line -> optional while under OPTIONAL_CAP
OPTIONAL_CAP = 0.25  # absorption is refused once optional lines would exceed this share of a node's line budget
WINDOW_COST_USD = 0.055

# logs_agent (M7, decisions 45, 48, 49): one failed run (case) -> RCA shortlist.
CASES_DIR = DATA / "cases"  # one shortlist JSON per case
CASE_PROMPT_PATH = ROOT_DIR / "matcher" / "prompts" / "segmentation_failure.txt"  # failure-tuned prompt
CASE_MODEL_ID = os.environ.get("CLEW_CASE_MODEL", MODEL_ID)  # model for failed cases (decision 49); defaults to CLEW_LLM_MODEL
CASE_TOKEN_CEILING = 50_000  # per-case input-token ceiling
DAILY_TOKEN_BUDGET = 500_000  # global per-flow-type budget per day, checked before Phase 3
BUDGET_PATH = DATA / "budget" / f"{FLOW_TYPE}.json"  # tokens spent today by logs_agent for this flow type
PROMOTE_MIN_HEALTHY = 3  # decision 48: a logs_agent node needs >= 3 healthy sightings to be promoted ...
# ... and its healthy share (healthy runs / training runs) must be >= its failed share (cases / all cases).  # ~cost of one 128-line window, measured on run 1 (for the estimate only)
