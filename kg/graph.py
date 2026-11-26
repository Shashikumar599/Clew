"""Stage 6: the graph -- sentinel root, per-stream sub-roots, nodes, and edges in run order.

Stream mode (decision 38): every service log file is its own stream. A single sentinel
root anchors one sub-root per stream ("ROOT: neutron/server"), and each stream's steps hang below
its sub-root. Edges only connect steps of the same stream, in that stream's time order.
A step that recurs in a stream is the same node revisited, so the edge back to it is a cross-link.
is_loop = the node has an edge to itself or to an earlier-created node (decision 29).
Time-overlap links between streams are counts on each node (`overlap_counts`), not edges.
"""
from datetime import datetime

from kg.node import new_node, part_node, root_node, service_root_node, widen


class Graph:
    def __init__(self, repo, flow_type):
        self.repo, self.flow_type = repo, flow_type
        root = root_node(repo.graph_id, flow_type)
        if repo.get(root["chunk_id"]) is None:
            repo.put(root)
        self.root_id = root["chunk_id"]

    def real_nodes(self):
        return [n for n in self.repo.all()
                if not n.get("is_root") and not n.get("is_service_root") and not n.get("superseded_by")]

    def service_root(self, service, create=True):
        """The stream's sub-root id (created and wired under the root on first use)."""
        sub = service_root_node(self.repo.graph_id, self.flow_type, service, seq=len(self.repo.all()))
        if self.repo.get(sub["chunk_id"]) is None:
            if not create:
                return None
            self.repo.put(sub)
            self.add_transition(self.root_id, sub["chunk_id"])
        return sub["chunk_id"]

    def visit(self, step, run_id, service=None):
        """Create the step's node, or widen the existing one. Returns (chunk_id, created)."""
        node = new_node(self.repo.graph_id, step, seq=len(self.repo.all()), run_id=run_id, service=service)
        existing = self.repo.get(node["chunk_id"])
        if existing is None:
            self.repo.put(node)
            return node["chunk_id"], True
        if existing.pop("superseded_by", None):  # a step that was split away comes back whole: revive it
            existing["revived"] = True
        widen(existing, step, run_id)
        return existing["chunk_id"], False

    def add_transition(self, parent_id, child_id):
        parent, child = self.repo.get(parent_id), self.repo.get(child_id)
        if child_id not in parent["child_chunk_ids"]:
            parent["child_chunk_ids"].append(child_id)
        if parent_id not in child["parent_chunk_ids"]:
            child["parent_chunk_ids"].append(parent_id)
        counts = parent["child_transition_counts"]
        counts[child_id] = counts.get(child_id, 0) + 1
        if child["seq"] <= parent["seq"]:  # back to itself or an earlier step
            parent["is_loop"] = True

    def record_overlaps(self, items):
        """items: (service, chunk_id, first_ts, last_ts) for every step of one run. Each pair of steps
        from different streams whose time spans overlap bumps both nodes' overlap_counts (streams are
        never linked by edges). Returns the number of overlapping pairs."""
        epoch = lambda ts: datetime.fromisoformat(ts).timestamp()
        spans = sorted((epoch(a), epoch(b), s, c) for s, c, a, b in items)
        active, pairs = [], 0
        for t0, t1, s, c in spans:
            active = [x for x in active if x[1] >= t0]  # drop steps that ended before this one starts
            for _, _, s2, c2 in active:
                if s2 != s:
                    for x, y in ((c, c2), (c2, c)):
                        oc = self.repo.get(x).setdefault("overlap_counts", {})
                        oc[y] = oc.get(y, 0) + 1
                    pairs += 1
            active.append((t0, t1, s, c))
        return pairs

    def split_node(self, nid, cut):
        """split_node(): a visit matched only a prefix of the node -> split it into matched + residual.
        N becomes N1 (templates before `cut`) -> N2 (the rest). N's parents now point to N1; N's
        children hang below N2; an edge back to N itself becomes N2 -> N1. The old node stays in the
        file marked superseded_by [N1, N2] with no edges (decision 14: no deletion). Returns (n1, n2)."""
        repo, old = self.repo, self.repo.get(nid)
        n1 = part_node(old, old["line_templates"][:cut], old["seq"], old["title"])
        n2 = part_node(old, old["line_templates"][cut:], old["seq"] + 0.5, old["title"] + " (continued)")
        for part in (n1, n2):
            if repo.get(part["chunk_id"]) is not None and not repo.get(part["chunk_id"]).get("superseded_by"):
                part["merged_into_existing"] = True  # same step already in the graph: reuse it (idempotent ids)
            else:
                repo.put(part)
        a, b = n1["chunk_id"], n2["chunk_id"]
        out_counts = old["child_transition_counts"]
        for p in old["parent_chunk_ids"]:  # incoming edges p -> N become p -> N1
            if p == nid:
                continue
            P = repo.get(p)
            c = P["child_transition_counts"].pop(nid, 0)
            P["child_chunk_ids"] = [x for x in P["child_chunk_ids"] if x != nid]
            self._link(p, a, c)
        for ch in old["child_chunk_ids"]:  # outgoing edges N -> c become N2 -> c (N -> N becomes N2 -> N1)
            target = a if ch == nid else ch
            if ch != nid:
                C = repo.get(ch)
                C["parent_chunk_ids"] = [x for x in C["parent_chunk_ids"] if x != nid]
            self._link(b, target, out_counts.get(ch, 0))
        self._link(a, b, max(1, sum(out_counts.values())))
        for n in repo.all():  # overlap links that pointed at N now point at N1
            oc = n.get("overlap_counts")
            if oc and nid in oc:
                oc[a] = oc.get(a, 0) + oc.pop(nid)
        old.update({"superseded_by": [a, b], "parent_chunk_ids": [], "child_chunk_ids": [],
                    "child_transition_counts": {}})
        return a, b

    def _link(self, parent_id, child_id, count):
        parent, child = self.repo.get(parent_id), self.repo.get(child_id)
        if child_id not in parent["child_chunk_ids"]:
            parent["child_chunk_ids"].append(child_id)
        if parent_id not in child["parent_chunk_ids"]:
            child["parent_chunk_ids"].append(parent_id)
        parent["child_transition_counts"][child_id] = parent["child_transition_counts"].get(child_id, 0) + count
        if child["seq"] <= parent["seq"]:
            parent["is_loop"] = True
