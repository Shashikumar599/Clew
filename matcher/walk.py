"""Walk one run through the graph, one STEP at a time (approach A, decision 35). Read-only.

Why steps, not lines: our nodes are multi-line chunks and most line templates appear in several
nodes, so "this line fits a template of node X" says little. A node is recognised by a run of
consecutive lines that together cover enough of it.

At a step boundary (pointer = last visited node):
  Phase 0: candidates = the pointer's children (+ the pointer itself, a repeat). For each, look
           ahead: consume upcoming lines while they fit the node (exact -> regex -> fuzzy per line).
           Accept only if the visit covers >= COVERAGE_MIN of the node's distinct templates.
           Once covered, the visit ends where a child of the node can start an accepted visit
           (handover), so a node does not swallow the next step's shared lines.
           Pick the longest accepted visit; tie -> the most-taken edge.
  Phase 1: no Phase 0 candidate accepted -> every node holding a template that fits this line
           (exact or fuzzy score) gets the same check; a hit moves the pointer (decision 33).
  Miss:    the line stays unmatched; the pointer stays.
Phase 2, absorption and Phase 3 are not run here: unmatched spans are the output.
"""
import math
from collections import defaultdict

from matcher.fuzzy import Line, Template, score, threshold


class GraphIndex:
    def __init__(self, repo, root_id, cfg):
        self.cfg, self.root_id = cfg, root_id
        self.nodes = {n["chunk_id"]: n for n in repo.all()}
        self.tpls = {}  # node_id -> [Template]
        self.exact = {}  # node_id -> {text: Template}
        self.nodes_with = defaultdict(set)  # template text -> node ids (Phase 1, exact)
        self.by_token = defaultdict(list)  # literal token -> [Template] (Phase 1, fuzzy pre-filter)
        for nid, n in self.nodes.items():
            ts = [Template(lt["template"], nid, cfg.PREFIX_TOKENS) for lt in n["line_templates"]]
            self.tpls[nid] = ts
            self.exact[nid] = {t.text: t for t in ts}
            for t in ts:
                self.nodes_with[t.text].add(nid)
                for tok in t.lit_set:
                    self.by_token[tok].append(t)
        self.children = {nid: sorted(n["child_chunk_ids"], key=lambda c: -n["child_transition_counts"].get(c, 0))
                         for nid, n in self.nodes.items()}

    def edge_count(self, a, b):
        return self.nodes[a]["child_transition_counts"].get(b, 0)


class Walker:
    def __init__(self, entries, idx):
        self.idx, self.cfg = idx, idx.cfg
        self.lines = [Line(e["template"], self.cfg.PREFIX_TOKENS) for e in entries]
        self.memo = {}

    def line_hit(self, nid, j, confirmed):
        """Does line j fit node nid? Tiers: exact -> regex -> fuzzy. Returns (template, tier, score)."""
        line, idx = self.lines[j], self.idx
        t = idx.exact[nid].get(line.text)
        if t:
            return t.text, "exact", 1.0
        for t in idx.tpls[nid]:
            if t.regex and t.regex.fullmatch(line.text):
                return t.text, "regex", 1.0
        th = threshold(line, len(confirmed) * 2 >= len(idx.tpls[nid]), self.cfg)
        best = None
        for t in idx.tpls[nid]:
            s = score(line, t)
            if s >= th and (best is None or s > best[2]):
                best = (t.text, "fuzzy", s)
        return best

    def extend(self, nid, j):
        """Consecutive lines from j that fit nid (memoised). The scan length is bounded only to keep
        the cost down (EXTEND_SCAN_*), not as a rule on how long a step may be."""
        key = (nid, j)
        if key not in self.memo:
            limit = self.idx.nodes[nid]["line_budget"]["max"] * self.cfg.EXTEND_SCAN_FACTOR + self.cfg.EXTEND_SCAN_EXTRA
            out, confirmed = [], set()
            while j + len(out) < len(self.lines) and len(out) < limit:
                h = self.line_hit(nid, j + len(out), confirmed)
                if not h:
                    break
                out.append(h)
                confirmed.add(h[0])
            self.memo[key] = out
        return self.memo[key]

    def accept(self, nid, j):
        """(hits, m): m = lines needed to reach the coverage guard; None if it is never reached."""
        hits = self.extend(nid, j)
        need = max(1, math.ceil(self.cfg.COVERAGE_MIN * len(self.idx.tpls[nid])))
        seen = set()
        for m, h in enumerate(hits, 1):
            seen.add(h[0])
            if len(seen) >= need:
                return hits, m
        return None

    def visit_length(self, nid, j, hits, m0):
        """Hand over to the first position where a child of nid can start a visit -- but only after
        the visit has seen every template it is going to see in the look-ahead (m_full), so a node
        is not cut before its own remaining lines."""
        seen, m_full = set(), 0
        for m, h in enumerate(hits, 1):
            if h[0] not in seen:
                seen.add(h[0])
                m_full = m
        for m in range(max(m0, m_full), len(hits)):
            if any(d != nid and self.accept(d, j + m) for d in self.idx.children[nid]):
                return m
        return len(hits)

    def best_visit(self, cands, j, rank):
        best = None
        for c in cands:
            a = self.accept(c, j)
            if a:
                length = self.visit_length(c, j, *a)
                key = (length, rank(c))
                if best is None or key > best[0]:
                    best = (key, c, a[0][:length])
        return best

    def phase1_candidates(self, j, ptr):
        line, idx = self.lines[j], self.idx
        cands = set(idx.nodes_with.get(line.text, ()))
        for tok in line.tok_set:
            for t in idx.by_token.get(tok, ()):
                if t.node_id not in cands and score(line, t) >= threshold(line, False, self.cfg):
                    cands.add(t.node_id)
        return cands

    def run(self):
        idx, i, n = self.idx, 0, len(self.lines)
        ptr, gap = idx.root_id, False
        lines, visits, transitions = [], [], []
        while i < n:
            cands = idx.children[ptr] + ([ptr] if ptr != idx.root_id else [])
            best, phase = self.best_visit(cands, i, lambda c: idx.edge_count(ptr, c)), "phase0"
            if best is None:
                best, phase = self.best_visit(self.phase1_candidates(i, ptr), i,
                                              lambda c: -idx.nodes[c]["seq"]), "phase1"
            if best is None:
                lines.append({"i": i, "phase": "unmatched"})
                i, gap = i + 1, True
                continue
            _, nid, hits = best
            transitions.append({"from": ptr, "to": nid, "after_gap": gap, "phase": phase})
            visit = {"node": nid, "lines": [], "counts": defaultdict(int), "phase": phase}
            for k, (tpl, tier, s) in enumerate(hits):
                visit["lines"].append(i + k)
                visit["counts"][tpl] += 1
                lines.append({"i": i + k, "phase": phase, "method": tier, "score": round(s, 3), "node": nid})
            visits.append(visit)
            ptr, i, gap = nid, i + len(hits), False
        return lines, visits, transitions


def match_run(entries, idx):
    return Walker(entries, idx).run()
