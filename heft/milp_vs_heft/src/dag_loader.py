from __future__ import annotations
import json
from typing import Dict, Any
from dataclasses import dataclass
from heft_week16.src.dag import DAG, Task, Edge  # if imports break later, copy dag.py locally

def load_dag_json(path: str) -> DAG:
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    devices = [d["id"] for d in obj["devices"]]
    tasks = {t["id"]: Task(t["id"], {k: float(v) for k,v in t["costs"].items()}) for t in obj["tasks"]}
    edges = [Edge(e["src"], e["dst"], int(e["bytes"])) for e in obj["edges"]]
    return DAG(devices=devices, tasks=tasks, edges=edges)
