"""Stage 5: turn one segmented step into a graph node (decisions 23, 27).

A node is a chunk: one logical step made of several consecutive lines.
- log_template = the step's DISTINCT line templates, in first-seen order, joined by newlines
  (repeats live in the per-line counts, not in the text).
- chunk_id = sha256(log_template), so the same step resolves to the same node from any run (doc).
- Each line template has a (min, max) repeat count; the node has a line budget (min, max total lines).
"""
import hashlib, time

ROOT_TITLE_PREFIX = "ROOT: "


def now_ms():
    return int(time.time() * 1000)


def chunk_of(lines):
    """Distinct templates in first-seen order + how often each appears in this occurrence."""
    counts = {}
    for e in lines:
        counts[e["template"]] = counts.get(e["template"], 0) + 1
    log_template = "\n".join(counts)  # dict keeps first-seen order
    return hashlib.sha256(log_template.encode()).hexdigest(), log_template, counts


def root_node(graph_id, flow_type):
    """Sentinel root: anchors every real step of the flow type (doc). Its id is a hash of a fixed
    text, so it is deterministic too (our choice; the doc does not say how the root is keyed)."""
    log_template = f"<ROOT {flow_type}>"
    return {
        "graph_id": graph_id, "chunk_id": hashlib.sha256(log_template.encode()).hexdigest(),
        "title": ROOT_TITLE_PREFIX + flow_type, "log_template": log_template, "is_root": True,
        "line_templates": [], "line_budget": {"min": 0, "max": 0},
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {},
        "is_loop": False, "seq": 0, "creator": "kg_builder",
        "run_count": 0, "access_count": 0, "created_at_epoch_ms": now_ms(), "last_accessed_at_epoch_ms": None,
    }


def new_node(graph_id, step, seq, run_id):
    """A node from a step the graph has never seen. Counts start at what this one occurrence showed."""
    chunk_id, log_template, counts = chunk_of(step["lines"])
    return {
        "graph_id": graph_id, "chunk_id": chunk_id,
        # LLM metadata (doc: summary, expected_behavior, severity, category, tags; + title, fix_suggestion)
        "title": step["title"], "summary": step["summary"], "expected_behavior": step["expected_behavior"],
        "severity": step["severity"], "category": step["category"], "tags": step["tags"],
        "fix_suggestion": step["fix_suggestion"],
        "log_template": log_template,
        "line_templates": [{"template": t, "min": c, "max": c} for t, c in counts.items()],
        "line_budget": {"min": len(step["lines"]), "max": len(step["lines"])},
        # adjacency list on the node (doc) + per-child transition counts (doc)
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {},
        "is_loop": False,
        "seq": seq,  # creation order; used for the is_loop rule (decision 29)
        "creator": "kg_builder",  # provenance (doc: kg_builder vs logs_agent)
        # telemetry: access_count / last_accessed_at_epoch_ms (doc); run_count per decision 14
        "run_count": 1, "last_run_id": run_id, "access_count": 1,
        "created_at_epoch_ms": now_ms(), "last_accessed_at_epoch_ms": now_ms(),
    }


def widen_counts(node, counts, n_lines, run_id):
    """A matched visit in a happy run (decisions 23, 31): widen each template's (min, max) by what this
    visit showed -- a template the visit did not use counts as 0, so its min drops to 0 -- and the
    node's line budget by the visit's line total. Template text, log_template and chunk_id do not change."""
    for lt in node["line_templates"]:
        c = counts.get(lt["template"], 0)
        lt["min"], lt["max"] = min(lt["min"], c), max(lt["max"], c)
    b = node["line_budget"]
    b["min"], b["max"] = min(b["min"], n_lines), max(b["max"], n_lines)
    if node.get("last_run_id") != run_id:
        node["run_count"] += 1
        node["last_run_id"] = run_id
    node["access_count"] += 1
    node["last_accessed_at_epoch_ms"] = now_ms()


def widen(node, step, run_id):
    """The same step seen again (same chunk_id): widen the counts, bump telemetry.
    LLM metadata stays as first written (our choice; the doc does not say which wins)."""
    _, _, counts = chunk_of(step["lines"])
    for lt in node["line_templates"]:  # same chunk_id => same set of templates
        c = counts[lt["template"]]
        lt["min"], lt["max"] = min(lt["min"], c), max(lt["max"], c)
    b = node["line_budget"]
    b["min"], b["max"] = min(b["min"], len(step["lines"])), max(b["max"], len(step["lines"]))
    if node.get("last_run_id") != run_id:
        node["run_count"] += 1
        node["last_run_id"] = run_id
    node["access_count"] += 1
    node["last_accessed_at_epoch_ms"] = now_ms()
