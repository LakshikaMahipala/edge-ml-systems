from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from pareto_core import pareto_front

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_all", type=str, default="results/codesign_points.jsonl")
    ap.add_argument("--out_pareto", type=str, default="results/pareto_points.jsonl")
    ap.add_argument("--out_selected", type=str, default="results/selected_points.json")
    args = ap.parse_args()

    pts = load_jsonl(args.in_all)
    front = pareto_front(pts)

    # write pareto jsonl
    out_p = Path(args.out_pareto)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in front:
            f.write(json.dumps(r) + "\n")

    # select representative points
    # fast = lowest latency
    fast = min(front, key=lambda r: r["latency"])
    # accurate = highest accuracy
    accurate = max(front, key=lambda r: r["acc"])
    # balanced = closest to middle tradeoff (normalize)
    lat_min, lat_max = min(r["latency"] for r in front), max(r["latency"] for r in front)
    acc_min, acc_max = min(r["acc"] for r in front), max(r["acc"] for r in front)
    ene_min, ene_max = min(r["energy"] for r in front), max(r["energy"] for r in front)

    def norm(x, a, b): 
        return 0.0 if b == a else (x - a) / (b - a)

    def balance_score(r):
        # want low latency/energy and high acc -> use distance to ideal (acc=1, lat=0, ene=0)
        acc_n = norm(r["acc"], acc_min, acc_max)
        lat_n = norm(r["latency"], lat_min, lat_max)
        ene_n = norm(r["energy"], ene_min, ene_max)
        # ideal: acc_n=1, lat_n=0, ene_n=0
        return (1 - acc_n)**2 + (lat_n)**2 + (ene_n)**2

    balanced = min(front, key=balance_score)

    selected = {
        "fast": fast,
        "balanced": balanced,
        "accurate": accurate,
        "pareto_front_size": len(front),
        "note": "All numbers are proxies until real FPGA measurements are plugged in.",
    }

    Path(args.out_selected).write_text(json.dumps(selected, indent=2))
    print("wrote:", args.out_pareto, "and", args.out_selected)
    print("pareto size:", len(front))

if __name__ == "__main__":
    main()
