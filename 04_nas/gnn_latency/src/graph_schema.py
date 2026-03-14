from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any

@dataclass
class Node:
    node_id: str
    op_type: str
    attrs: Dict[str, Any]  # kernel, stride, shapes, dtype, etc.

@dataclass
class Edge:
    src: str
    dst: str
    attrs: Dict[str, Any]  # tensor_bytes, layout, etc.

@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]
    globals: Dict[str, Any]  # device specs, batch, runtime flags
