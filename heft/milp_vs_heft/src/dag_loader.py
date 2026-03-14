from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Task:
    id: str
    costs: Dict[str, float]

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
            raise ValueError("DAG has a cycle")
        return out

def load_dag_json(path: str) -> DAG:
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    devices = [d["id"] for d in obj["devices"]]
    tasks = {t["id"]: Task(t["id"], {k: float(v) for k,v in t["costs"].items()}) for t in obj["tasks"]}
    edges = [Edge(e["src"], e["dst"], int(e["bytes"])) for e in obj["edges"]]
    return DAG(devices=devices, tasks=tasks, edges=edges)
