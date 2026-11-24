"""Text scoring shared by Phase 0 (fuzzy tier) and Phase 1.

Doc: "a blend of token-set containment (with placeholders stripped from the template side) and a
prefix-agreement score over the first six literal tokens". The threshold "decays for very long
lines, and relaxes the more node lines have already been confirmed matched in the current span".
Exact formula and numbers are ours (decision 34).
"""
import re

PLACEHOLDER = re.compile(r"<(?:\*|[A-Z0-9_]+)>")  # <*>, <UUID>, <IP>, ...


def literal(tokens):
    return [t for t in tokens if not PLACEHOLDER.fullmatch(t)]


class Line:
    """A run's line template, pre-split once."""
    __slots__ = ("text", "tokens", "tok_set", "prefix")

    def __init__(self, text, prefix_n):
        self.text = text
        self.tokens = text.split()
        self.tok_set = set(self.tokens)
        self.prefix = literal(self.tokens)[:prefix_n]


class Template:
    """One line template of a graph node, with what the matchers need precomputed."""
    __slots__ = ("text", "node_id", "lit_set", "prefix", "regex")

    def __init__(self, text, node_id, prefix_n):
        self.text, self.node_id = text, node_id
        lit = literal(text.split())
        self.lit_set = set(lit)
        self.prefix = lit[:prefix_n]
        # Phase 0 regex tier: "a regex compiled from the node's placeholder pattern".
        # Each placeholder becomes a wildcard, so a line that kept a literal value still matches.
        parts = PLACEHOLDER.split(text)
        self.regex = re.compile(".+?".join(re.escape(p) for p in parts)) if len(parts) > 1 else None


def score(line, tpl):
    """Average of token containment (template literals found in the line) and prefix agreement."""
    if not tpl.lit_set:
        return 0.0
    containment = len(tpl.lit_set & line.tok_set) / len(tpl.lit_set)
    k = len(tpl.prefix)
    prefix = sum(a == b for a, b in zip(tpl.prefix, line.prefix)) / k if k else 0.0
    return (containment + prefix) / 2


def threshold(line, relaxed, cfg):
    th = cfg.FUZZY_BASE
    extra = len(line.tokens) - cfg.FUZZY_LONG_LINE
    if extra > 0:  # long lines: decays per extra token, down to the floor
        th = max(cfg.FUZZY_FLOOR, th - cfg.FUZZY_DECAY_PER_TOKEN * extra)
    if relaxed:  # half of the node's templates already confirmed in this visit
        th -= cfg.FUZZY_RELAX
    return th
