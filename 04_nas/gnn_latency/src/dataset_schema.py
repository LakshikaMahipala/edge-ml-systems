from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from graph_schema import Graph

@dataclass
class LabeledGraph:
    graph: Graph
    target: Dict[str, float]  # {"latency_p50_ms": ...}
    meta: Dict[str, Any]      # device, runtime, precision, etc.
