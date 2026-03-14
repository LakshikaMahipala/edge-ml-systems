from __future__ import annotations
import argparse
import json

from dag import DAG, Task, Edge
from comm import LinkModel
from heft import heft_schedule, makespan
from export_schedule import export_json

def load_dag(path: str) -> DAG:
    obj = json.loads(open(path, "r", encoding="utf-8").read())
    devices = [d["id"] for d in obj["devices"]]
    tasks = {t["id"]: Task(t["id"], {k: float(v) for k,v in t["costs"].items()}) for t in obj["tasks"]}
    edges = [Edge(e["src"], e["dst"], int(e["bytes"])) for e in obj["edges"]]
    return DAG(devices=devices, tasks=tasks, edges=edges)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dag", type=str, default="schema/example_dag.json")
    ap.add_argument("--bw_GBps", type=float, default=10.0)
    ap.add_argument("--overhead_ms", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="results/schedule.json")
    args = ap.parse_args()

    dag = load_dag(args.dag)
    link = LinkModel(bandwidth_GBps=args.bw_GBps, overhead_ms=args.overhead_ms)
    sched = heft_schedule(dag, link)

    export_json(args.out, sched)
    print("makespan_ms:", makespan(sched))
    print("wrote:", args.out)

if __name__ == "__main__":
    main()
