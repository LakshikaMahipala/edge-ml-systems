from __future__ import annotations
from typing import Dict, Any
from graph_schema import Graph

def extract_node_features(node_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub: convert raw node attrs into numeric features.
    Later:
    - one-hot op_type
    - normalize shapes
    - compute MACs, bytes, intensity
    """
    return node_attrs

def extract_edge_features(edge_attrs: Dict[str, Any]) -> Dict[str, Any]:
    return edge_attrs

def featurize_graph(g: Graph) -> Dict[str, Any]:
    return {
        "nodes": [extract_node_features(n.attrs | {"op_type": n.op_type}) for n in g.nodes],
        "edges": [extract_edge_features(e.attrs | {"src": e.src, "dst": e.dst}) for e in g.edges],
        "globals": g.globals,
    }
