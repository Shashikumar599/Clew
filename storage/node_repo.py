"""Stage 7: NodeRepo -- one local JSON file per graph (decision 26).

The file is a map chunk_id -> node, each node in the DynamoDB item shape (keys graph_id + chunk_id),
so swapping this class for DynamoDB Local later (M1) keeps the same get / put / children calls.
Difference from DynamoDB: puts stay in memory until flush(), which writes the whole file at once.
"""
import json, os


class NodeRepo:
    def __init__(self, graph_dir, graph_id):
        self.graph_id = graph_id
        self.path = graph_dir / f"{graph_id}.json"
        self.nodes = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, chunk_id):
        return self.nodes.get(chunk_id)

    def put(self, node):
        assert node["graph_id"] == self.graph_id
        self.nodes[node["chunk_id"]] = node

    def children(self, chunk_id):
        """Point-reads of a node's children (the adjacency list lives on the node, as in the doc)."""
        return [self.nodes[c] for c in self.nodes[chunk_id]["child_chunk_ids"]]

    def all(self):
        return list(self.nodes.values())

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.nodes, indent=1))
        os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written graph
