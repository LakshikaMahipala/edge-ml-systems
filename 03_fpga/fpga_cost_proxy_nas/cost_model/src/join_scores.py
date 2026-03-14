from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_fpga", type=str, default="results/ranked.jsonl")
    ap.add_argument("--reduced_proxy", type=str, default="../nas_proxies/reduced_training_harness/results/proxy_scores.jsonl")
    ap.add_argument("--zero_cost", type=str, default="../nas_zero_cost/zero_cost_harness/results/zero_cost.jsonl")
    ap.add_argument("--out", type=str, default="results/joined.jsonl")
    args = ap.parse_args()

    fpga = {r["arch_id"]: r for r in load_jsonl(args.ranked_fpga)}
    red = [r for r in load_jsonl(args.reduced_proxy) if r.get("type") == "proxy"]
    red = {r["arch_id"]: r for r in red}
    zc = {r["arch_id"]: r for r in load_jsonl(args.zero_cost)}

    keys = sorted(set(fpga) | set(red) | set(zc))
    out_rows = []
    for k in keys:
        out_rows.append({
            "arch_id": k,
            "final_score_fpga": fpga.get(k, {}).get("final_score", None),
            "hw_cost_sum": fpga.get(k, {}).get("hw_cost_sum", None),
            "acc_proxy_fpga": fpga.get(k, {}).get("acc_proxy", None),
            "reduced_train_score": red.get(k, {}).get("score", None),
            "zero_cost_snip": zc.get(k, {}).get("snip_score", None),
            "zero_cost_gradnorm": zc.get(k, {}).get("gradnorm_score", None),
        })

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print("wrote:", out_p)

if __name__ == "__main__":
    main()
