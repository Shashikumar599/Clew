"""Text scoring shared by Phase 0 (fuzzy tier) and Phase 1.

Score = blend of token-set containment (placeholders stripped from the template side) and a
prefix-agreement score over the first six literal tokens. The threshold decays for very long
lines and relaxes once more of the node's lines are confirmed in the current visit (decision 34).
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
    __slots__ = ("text", "node_id", "lit_set", "prefix", "pattern")

    def __init__(self, text, node_id, prefix_n):
        self.text, self.node_id = text, node_id
        tokens = text.split()
        lit = literal(tokens)
        self.lit_set = set(lit)
        self.prefix = lit[:prefix_n]
        # Phase 0 regex tier: "a regex compiled from the node's placeholder pattern".
        # Matched token by token (a Drain3 <*> stands for one token): per token, None = must be
        # equal, True = whole-token placeholder (anything), or a small regex for a mixed token
        # such as req-<UUID>. A whole-line regex with many .+? backtracked for seconds per line.
        if PLACEHOLDER.search(text):
            self.pattern = [True if PLACEHOLDER.fullmatch(t) else
                            re.compile("[^ ]+?".join(re.escape(p) for p in PLACEHOLDER.split(t)))
                            if PLACEHOLDER.search(t) else None for t in tokens]
        else:
            self.pattern = None

    def regex_match(self, line):
        """Same token count, and every token equal or fitting its placeholder."""
        if self.pattern is None or len(line.tokens) != len(self.pattern):
            return False
        for p, want, got in zip(self.pattern, self.text.split(), line.tokens):
            if p is None:
                if want != got:
                    return False
            elif p is not True and not p.fullmatch(got):
                return False
        return True


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
