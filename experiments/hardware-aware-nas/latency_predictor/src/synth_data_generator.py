from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

def gen_graph(i: int, rng: random.Random):
    # random "conv graph" with a few ops
    n = rng.randint(4, 10)
    nodes = []
    for j in range(n):
        k = rng.choice([3,5])
        cin = rng.choice([8,16,24,32])
        cout = rng.choice([8,16,24,32])
        h = w = rng.choice([32,16,8])
        nodes.append({"node_id": f"n{j}", "op_type": "conv", "attrs": {"cin": cin, "cout": cout, "h": h, "w": w, "k": k, "stride": 1}})
    # synthetic latency: weighted by macs + bytes
    macs = 0.0
    bytes_m = 0.0
    for nd in nodes:
        a = nd["attrs"]
        macs += a["cout"]*a["h"]*a["w"]*a["cin"]*(a["k"]**2)
        bytes_m += (a["cin"]*a["h"]*a["w"] + a["cout"]*a["h"]*a["w"])
    latency = 0.2 + 1e-9*macs + 1e-6*bytes_m + rng.uniform(-0.02, 0.02)

    return {
        "graph_id": f"G{i:05d}",
        "globals": {"batch": 1, "device": "proxy", "precision": "int8"},
        "nodes": nodes,
        "edges": [],  # unused in baseline
        "target": {"latency_p50_ms": float(max(latency, 0.001))}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/graph_latency.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            f.write(json.dumps(gen_graph(i, rng)) + "\n")

    print("wrote:", out_p)

if __name__ == "__main__":
    main()
