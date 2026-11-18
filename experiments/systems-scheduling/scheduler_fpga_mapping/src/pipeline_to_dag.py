from __future__ import annotations
import json
from typing import Dict, Any
from pathlib import Path

from latency_registry import load_registry, build_cost_table, ensure_all_devices

def build_dag(pipeline_spec_path: str, registry_path: str, out_path: str):
    spec = json.loads(open(pipeline_spec_path, "r", encoding="utf-8").read())
    entries = load_registry(registry_path)
    cost_table = build_cost_table(entries)

    devices = spec["devices"]
    ensure_all_devices(cost_table, devices)

    tasks = []
    for t in spec["tasks"]:
        kid = t["kernel_id"]
        costs = {d: float(cost_table[kid][d]) for d in devices}
        tasks.append({"id": t["id"], "costs": costs})

    dag = {
        "devices": [{"id": d} for d in devices],
        "tasks": tasks,
        "edges": spec["edges"],
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(dag, indent=2))
    return out_path

def main():
    out = build_dag(
        "schema/example_pipeline_spec.json",
        "schema/example_latency_registry.json",
        "results/pipeline_dag.json"
    )
    print("wrote:", out)

if __name__ == "__main__":
    main()
