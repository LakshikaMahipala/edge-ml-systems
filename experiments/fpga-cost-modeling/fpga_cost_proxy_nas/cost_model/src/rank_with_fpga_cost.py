from __future__ import annotations
import argparse
import json
import math
from typing import Dict, Any, List
from pathlib import Path

from fpga_cost_model import estimate_fpga_cost

def acc_proxy_from_ops_params(ops: float, params: float) -> float:
    # saturating capacity proxy
    return float(1.0 - math.exp(-0.00000002 * ops - 0.000002 * params))

def approx_params_from_arch(arch: Dict[str, Any]) -> float:
    # very crude param proxy (enough for ranking demonstration)
    p = 0.0
    cin = 3
    stem = int(arch["stem_channels"])
    p += cin * stem * 3 * 3
    cin = stem
    for st in arch["stages"]:
        depth = int(st["depth"])
        cout = int(st["out_ch"])
        block = st["block"]
        k = int(st["k"])
        exp = int(st["exp"])
        se = int(st["se"])
        for _ in range(depth):
            if block == "conv":
                p += cin * cout * k * k
                cin = cout
            else:
                mid = cin * exp
                p += cin * mid       # pw1
                p += mid * k * k     # dw
                if se:
                    p += mid * 2
                p += mid * cout      # pw2
                cin = cout
    head = int(arch["head_channels"])
    p += cin * head
    p += head * int(arch["num_classes"])
    return float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="../nas_foundations/tiny_cnn_search_space/results/candidates.jsonl")
    ap.add_argument("--out", type=str, default="results/ranked.jsonl")
    ap.add_argument("--input_hw", type=int, default=32)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--lam_hw", type=float, default=0.25)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            arch_id = rec["arch_id"]
            arch = rec["arch"]

            cost = estimate_fpga_cost(arch, input_hw=args.input_hw, P=args.P).to_dict()
            params_p = approx_params_from_arch(arch)
            acc_p = acc_proxy_from_ops_params(cost["ops_total"], params_p)

            hw_cost = cost["cycles_proxy"] + cost["lut_proxy"] + cost["bram_proxy"] + cost["bw_proxy"]
            final = acc_p - args.lam_hw * hw_cost

            rows.append({
                "arch_id": arch_id,
                "acc_proxy": float(acc_p),
                "params_proxy": float(params_p),
                "fpga_cost": cost,
                "hw_cost_sum": float(hw_cost),
                "final_score": float(final),
                "arch": arch,
            })

    rows.sort(key=lambda r: r["final_score"], reverse=True)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("wrote:", out_p)
    if rows:
        print("top1 final_score:", rows[0]["final_score"], "arch_id:", rows[0]["arch_id"])

if __name__ == "__main__":
    main()
