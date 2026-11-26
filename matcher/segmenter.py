"""Stage 4: LLM segmentation of a run into logical steps (decisions 20-22).

The run's processed lines are sent in windows of WINDOW_LINES. In a non-final window the model ends
its last step at a clean boundary, and the lines after that step are carried into the next window
(non-final window hint; decision 22). Each window also gets the title of the step just
before it (decision 21).
"""
import json

SEVERITIES = {"info", "warn", "error"}
FIELDS = ["title", "summary", "expected_behavior", "severity", "category", "tags", "fix_suggestion"]

WINDOW_NOTE = {
    True: "This is the final window of the run: no lines follow it.",
    False: "This is NOT the final window: more lines of the run follow after it.",
}
BOUNDARY_RULE = {
    True: "- This is the final window: your steps must cover every line, up to the last one.",
    False: ("- This is not the final window: end your last step at a clean boundary instead of splitting an "
            "operation in the middle. Lines after your last step are sent again at the start of the next "
            "window, so you may leave a trailing unfinished operation out."),
}


def build_prompt(template, lines, final, before_title, after_title, service=None):
    """Fill the prompt placeholders. str.replace, not str.format: the prompt holds literal JSON braces."""
    numbered = "\n".join(f"{i} | {e['service']} | {e['template']}" for i, e in enumerate(lines, 1))
    return (template.replace("{service}", service or "")
            .replace("{window_note}", WINDOW_NOTE[final])
            .replace("{boundary_rule}", BOUNDARY_RULE[final])
            .replace("{before_title}", before_title or "none")
            .replace("{after_title}", after_title or "none")
            .replace("{lines}", numbered))


def parse_steps(reply, n_lines, final):
    """Parse and check the model's JSON. Raises ValueError with a reason the model can act on."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in the reply")
    try:
        steps = json.loads(reply[start:end + 1])["steps"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"reply is not the expected JSON ({e})")
    if not steps:
        raise ValueError("no steps returned")
    expect = 1  # steps must cover consecutive line ranges, in order, without gaps or overlaps
    for s in steps:
        missing = [k for k in ["start", "end"] + FIELDS if k not in s]
        if missing:
            raise ValueError(f"step starting at line {s.get('start')} is missing {missing}")
        if s["start"] != expect or not (s["start"] <= s["end"] <= n_lines):
            raise ValueError(f"step {s['start']}-{s['end']} breaks the rule: next step must start at line "
                             f"{expect} and end at or before line {n_lines}")
        if s["severity"] not in SEVERITIES:
            raise ValueError(f"step {s['start']}-{s['end']} has severity {s['severity']!r}, not one of {sorted(SEVERITIES)}")
        expect = s["end"] + 1
    if final and expect != n_lines + 1:
        raise ValueError(f"final window: steps end at line {expect - 1}, but must cover all {n_lines} lines")
    return steps


def segment_run(entries, llm, prompt_template, window_lines, max_attempts, max_windows=None, log=print,
                before_title=None, after_title=None, service=None):
    """Split a run (or one unmatched span of it) into steps. Returns (steps, stats).
    Each step = the model's metadata + `lines` (the entries it covers, in order).
    before_title / after_title: the known steps around a span (Phase 3 context)."""
    steps, stats = [], {"windows": 0, "input_tokens": 0, "output_tokens": 0, "retries": 0}
    pos, carry = 0, []  # pos = next unsent entry; carry = lines left after the last step of the previous window
    while carry or pos < len(entries):
        if max_windows is not None and stats["windows"] >= max_windows:
            log(f"  stopped after {max_windows} windows (--max-windows)")
            break
        take = window_lines - len(carry)
        window = carry + entries[pos:pos + take]
        pos += take
        final = pos >= len(entries)
        before = steps[-1]["title"] if steps else before_title
        base_prompt = build_prompt(prompt_template, window, final, before, after_title if final else None, service)
        prompt = base_prompt

        for attempt in range(1, max_attempts + 1):
            reply, usage = llm.complete(prompt)
            stats["input_tokens"] += usage.get("input_tokens", 0)
            stats["output_tokens"] += usage.get("output_tokens", 0)
            try:
                window_steps = parse_steps(reply, len(window), final)
                break
            except ValueError as e:
                if attempt == max_attempts:
                    err = RuntimeError(f"window {stats['windows'] + 1}: invalid reply after {attempt} attempts: {e}")
                    err.stats = stats  # tokens already spent, so the caller can still count them
                    raise err
                stats["retries"] += 1
                log(f"  window {stats['windows'] + 1}: invalid reply ({e}), asking again")
                prompt += f"\n\nNOTE: a previous answer to this request was rejected: {e}. Follow the rules exactly."
        if hasattr(llm, "store") and not usage.get("cached"):
            llm.store(base_prompt, reply, usage)  # keep only accepted replies, keyed by the original prompt

        for s in window_steps:
            steps.append({**{k: s[k] for k in FIELDS}, "lines": window[s["start"] - 1:s["end"]]})
        carry = window[window_steps[-1]["end"]:]  # non-final window: lines after the last step go round again
        stats["windows"] += 1
        log(f"  window {stats['windows']:>3}: {len(window):>3} lines -> {len(window_steps):>2} steps, "
            f"carry {len(carry):>3}{'  (cached)' if usage.get('cached') else ''}")
    return steps, stats
