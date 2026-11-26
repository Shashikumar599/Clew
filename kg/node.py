"""Stage 5: turn one segmented step into a graph node (decisions 23, 27).

A node is a chunk: one logical step made of several consecutive lines.
- log_template = the step's DISTINCT line templates, in first-seen order, joined by newlines
  (repeats live in the per-line counts, not in the text).
- chunk_id = sha256(log_template), so the same step resolves to the same node from any run.
- Each line template has a (min, max) repeat count; the node has a line budget (min, max total lines).
"""
import hashlib, time

ROOT_TITLE_PREFIX = "ROOT: "


def now_ms():
    return int(time.time() * 1000)


def chunk_of(lines, service=None):
    """Distinct templates in first-seen order + how often each appears in this occurrence.
    With a service (stream mode, decision 38) the service is part of the hashed text, so the same
    template set in two streams stays two nodes."""
    counts = {}
    for e in lines:
        counts[e["template"]] = counts.get(e["template"], 0) + 1
    log_template = "\n".join(counts)  # dict keeps first-seen order
    key = f"{service}\n{log_template}" if service else log_template
    return hashlib.sha256(key.encode()).hexdigest(), log_template, counts


def service_root_node(graph_id, flow_type, service, seq):
    """Sub-root of one stream (decision 38): ROOT -> ROOT: <service> -> that service's steps."""
    log_template = f"<ROOT {flow_type}/{service}>"
    return {
        "graph_id": graph_id, "chunk_id": hashlib.sha256(log_template.encode()).hexdigest(),
        "title": ROOT_TITLE_PREFIX + service, "log_template": log_template, "is_service_root": True,
        "service": service, "line_templates": [], "line_budget": {"min": 0, "max": 0},
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {},
        "is_loop": False, "seq": seq, "creator": "kg_builder",
        "run_count": 0, "access_count": 0, "created_at_epoch_ms": now_ms(), "last_accessed_at_epoch_ms": None,
    }


def root_node(graph_id, flow_type):
    """Sentinel root: anchors every real step of the flow type. Its id is a hash of a fixed
    text, so it is deterministic too."""
    log_template = f"<ROOT {flow_type}>"
    return {
        "graph_id": graph_id, "chunk_id": hashlib.sha256(log_template.encode()).hexdigest(),
        "title": ROOT_TITLE_PREFIX + flow_type, "log_template": log_template, "is_root": True,
        "line_templates": [], "line_budget": {"min": 0, "max": 0},
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {},
        "is_loop": False, "seq": 0, "creator": "kg_builder",
        "run_count": 0, "access_count": 0, "created_at_epoch_ms": now_ms(), "last_accessed_at_epoch_ms": None,
    }


def new_node(graph_id, step, seq, run_id, service=None):
    """A node from a step the graph has never seen. Counts start at what this one occurrence showed."""
    chunk_id, log_template, counts = chunk_of(step["lines"], service)
    return {
        "graph_id": graph_id, "chunk_id": chunk_id, "service": service,
        # steps of other streams that overlapped this one in time: chunk_id -> count (decision 38)
        "overlap_counts": {},
        # LLM metadata: title, summary, expected_behavior, severity, category, tags, fix_suggestion
        "title": step["title"], "summary": step["summary"], "expected_behavior": step["expected_behavior"],
        "severity": step["severity"], "category": step["category"], "tags": step["tags"],
        "fix_suggestion": step["fix_suggestion"],
        "log_template": log_template,
        "line_templates": [{"template": t, "min": c, "max": c} for t, c in counts.items()],
        "line_budget": {"min": len(step["lines"]), "max": len(step["lines"])},
        # adjacency list on the node + per-child transition counts
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {},
        "is_loop": False,
        "seq": seq,  # creation order; used for the is_loop rule (decision 29)
        "creator": "kg_builder",  # provenance: kg_builder or logs_agent
        # telemetry: access_count / last_accessed_at_epoch_ms; run_count per decision 14
        "run_count": 1, "last_run_id": run_id, "access_count": 1,
        "created_at_epoch_ms": now_ms(), "last_accessed_at_epoch_ms": now_ms(),
    }


def widen_counts(node, counts, n_lines, run_id, loosen=True, cap=None):
    """A matched visit in a happy run (decisions 23, 31): widen each template's (min, max) by what this
    visit showed, and the node's line budget by the visit's line total.
    loosen=True (v1-v3): a template the visit did not use counts as 0, so its min drops to 0.
    loosen=False (decision 40): a missing template keeps its min; only its `missed` counter goes up.
    Lines become optional only through absorb_line().
    loosen=False with cap (decision 41): a missing required line becomes optional like an absorbed extra
    line -- only while the node's optional share stays within `cap`; otherwise it stays required.
    Returns (loosened, refused) counts. Template text, log_template, chunk_id unchanged."""
    loosened = refused = 0
    for lt in node["line_templates"]:
        c = counts.get(lt["template"], 0)
        if c == 0 and not loosen:
            lt["missed"] = lt.get("missed", 0) + 1
            if cap is not None and lt["min"] > 0:
                if optional_share(node, extra=lt["max"], extra_in_budget=False) <= cap:
                    lt["min"] = 0
                    loosened += 1
                else:
                    refused += 1
            continue
        lt["min"], lt["max"] = min(lt["min"], c), max(lt["max"], c)
    b = node["line_budget"]
    b["min"], b["max"] = min(b["min"], n_lines), max(b["max"], n_lines)
    if node.get("last_run_id") != run_id:
        node["run_count"] += 1
        node["last_run_id"] = run_id
    node["access_count"] += 1
    node["last_accessed_at_epoch_ms"] = now_ms()
    return loosened, refused


def widen(node, step, run_id):
    """The same step seen again (same chunk_id): widen the counts, bump telemetry.
    LLM metadata stays as first written."""
    _, _, counts = chunk_of(step["lines"])
    for lt in node["line_templates"]:  # same chunk_id => same core templates; absorbed lines (min 0) may be absent
        c = counts.get(lt["template"], 0)
        lt["min"], lt["max"] = min(lt["min"], c), max(lt["max"], c)
    b = node["line_budget"]
    b["min"], b["max"] = min(b["min"], len(step["lines"])), max(b["max"], len(step["lines"]))
    if node.get("last_run_id") != run_id:
        node["run_count"] += 1
        node["last_run_id"] = run_id
    node["access_count"] += 1
    node["last_accessed_at_epoch_ms"] = now_ms()


def optional_share(node, extra=0, extra_in_budget=True):
    """Share of the node's line budget taken by optional (min=0) lines, on max counts.
    extra = lines that would become optional: new lines (absorb, also added to the budget) or
    existing required lines turning optional (decision 41, already inside the budget)."""
    opt = sum(lt["max"] for lt in node["line_templates"] if lt["min"] == 0) + extra
    return opt / max(1, node["line_budget"]["max"] + (extra if extra_in_budget else 0))


def absorb_line(node, template, cap):
    """absorb_optional_log_line(): a stray single line becomes an optional (min=0) line of the
    preceding node, refused when optional lines would exceed `cap` of the node's line budget.
    Returns "absorbed", "known" (template already on the node) or "refused".
    The absorbed line is NOT added to log_template, so chunk_id stays."""
    if any(lt["template"] == template for lt in node["line_templates"]):
        return "known"
    if optional_share(node, extra=1) > cap:
        return "refused"
    node["line_templates"].append({"template": template, "min": 0, "max": 0, "absorbed": True})
    return "absorbed"  # the visit's widen_counts() then sets max from the line count


def prefix_cut(node, counts):
    """split_node() trigger: only a prefix of the node matched. Prefix = the node's REQUIRED (min>0) templates in
    order; the visit used the first k of them and none after. Returns the position in line_templates
    of the first missing required template, or None if the visit is not prefix-shaped."""
    req = [i for i, lt in enumerate(node["line_templates"]) if lt["min"] > 0]
    seen = [i for i in req if node["line_templates"][i]["template"] in counts]
    if 0 < len(seen) < len(req) and seen == req[:len(seen)]:
        return req[len(seen)]
    return None


def partial_shape(node, counts):
    """prefix / suffix / middle / scattered / full -- which required templates a visit used (report only)."""
    req = [i for i, lt in enumerate(node["line_templates"]) if lt["min"] > 0]
    seen = [k for k, i in enumerate(req) if node["line_templates"][i]["template"] in counts]
    if len(seen) == len(req):
        return "full"
    if not seen:
        return "optional_only"
    if seen == list(range(len(seen))):
        return "prefix"
    if seen == list(range(len(req) - len(seen), len(req))):
        return "suffix"
    return "middle" if seen == list(range(seen[0], seen[0] + len(seen))) else "scattered"


def part_node(node, templates, seq, title):
    """One half of a split node: same metadata, its own templates, chunk_id from its own log_template."""
    core = [lt["template"] for lt in templates if not lt.get("absorbed")]
    log_template = "\n".join(core)
    svc = node.get("service")
    key = f"{svc}\n{log_template}" if svc else log_template
    part = {k: v for k, v in node.items()}
    part.update({
        "chunk_id": hashlib.sha256(key.encode()).hexdigest(), "log_template": log_template, "title": title,
        "line_templates": [dict(lt) for lt in templates],
        "line_budget": {"min": sum(lt["min"] for lt in templates), "max": sum(lt["max"] for lt in templates)},
        "parent_chunk_ids": [], "child_chunk_ids": [], "child_transition_counts": {}, "is_loop": False,
        "overlap_counts": dict(node.get("overlap_counts", {})), "seq": seq, "split_from": node["chunk_id"],
        "created_at_epoch_ms": now_ms(),
    })
    part.pop("superseded_by", None)
    return part


def new_case_node(graph_id, step, seq, case_id, service):
    """A node written by logs_agent's Phase 3 on a failed run (creator logs_agent: discovered on a failure).
    run_count keeps counting HEALTHY runs (0 here); failed_run_count counts cases (decision 48)."""
    node = new_node(graph_id, step, seq, run_id=None, service=service)
    node.update({"creator": "logs_agent", "run_count": 0, "last_run_id": None,
                 "failed_run_count": 1, "last_failed_case": case_id})
    return node
