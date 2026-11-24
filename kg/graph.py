"""Stage 6: the graph -- sentinel root, nodes, and edges in run order (decision 29).

A run is a path from the root through its steps. Each transition prev -> next adds an edge
(child on prev, parent on next) and bumps prev's per-child transition count. A step that recurs
in the run is the same node revisited, so the edge back to it is a cross-link.
is_loop = the node has an edge to itself or to an earlier-created node (our rule, decision 29).
"""
from kg.node import new_node, root_node, widen


class Graph:
    def __init__(self, repo, flow_type):
        self.repo = repo
        root = root_node(repo.graph_id, flow_type)
        if repo.get(root["chunk_id"]) is None:
            repo.put(root)
        self.root_id = root["chunk_id"]

    def real_nodes(self):
        return [n for n in self.repo.all() if not n.get("is_root")]

    def visit(self, step, run_id):
        """Create the step's node, or widen the existing one. Returns (chunk_id, created)."""
        node = new_node(self.repo.graph_id, step, seq=len(self.repo.all()), run_id=run_id)
        existing = self.repo.get(node["chunk_id"])
        if existing is None:
            self.repo.put(node)
            return node["chunk_id"], True
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
