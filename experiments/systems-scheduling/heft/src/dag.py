from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

@dataclass(frozen=True)
class Task:
    id: str
    costs: Dict[str, float]  # device_id -> time

@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    bytes: int

@dataclass
class DAG:
    devices: List[str]
    tasks: Dict[str, Task]
    edges: List[Edge]

    def parents(self, t: str) -> List[Edge]:
        return [e for e in self.edges if e.dst == t]

    def children(self, t: str) -> List[Edge]:
        return [e for e in self.edges if e.src == t]

    def topo_sort(self) -> List[str]:
        indeg = {tid: 0 for tid in self.tasks}
        for e in self.edges:
            indeg[e.dst] += 1
        q = [tid for tid, d in indeg.items() if d == 0]
        out = []
        while q:
            u = q.pop(0)
            out.append(u)
            for e in self.children(u):
                indeg[e.dst] -= 1
                if indeg[e.dst] == 0:
                    q.append(e.dst)
        if len(out) != len(self.tasks):
            raise ValueError("DAG has a cycle or disconnected issue")
        return out
