from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import math

from parse_measurements import load_jsonl
from predict_latency import predict

def ranks(items: Dict[str, float], higher_better: bool = False) -> Dict[str, float]:
    # for latency: lower is better
    keys = sorted(items.keys(), key=lambda k: items[k], reverse=higher_better)
    return {k: float(i + 1) for i, k in enumerate(keys)}

def spearman(r1: Dict[str, float], r2: Dict[str, float]) -> float:
    keys = sorted(set(r1) & set(r2))
    n = len(keys)
    if n < 2:
        return 0.0
    d2 = 0.0
    for k in keys:
        d = r1[k] - r2[k]
        d2 += d * d
    return 1.0 - (6.0 * d2) / (n * (n*n - 1.0))

def mape(pred: Dict[str, float], meas: Dict[str, float]) -> float:
    keys = sorted(set(pred) & set(meas))
    if not keys:
        return 0.0
    s = 0.0
    for k in keys:
        s += abs(pred[k] - meas[k]) / max(meas[k], 1e-9)
    return s / len(keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", type=str, default="data/measurements.jsonl")
    ap.add_argument("--out", type=str, default="data/validation.json")
    ap.add_argument("--w_table", type=float, default=1.0)
    ap.add_argument("--w_cycles", type=float, default=1.0)
    args = ap.parse_args()

    rows = load_jsonl(args.meas)

    pred = {}
    meas = {}
    meta = {}

    for r in rows:
        cid = r["case_id"]
        meas[cid] = float(r["latency_p50_us"])
        pred[cid] = float(predict(r, w_table=args.w_table, w_cycles=args.w_cycles))
        meta[cid] = {"op": r["op"], "shape": [r["cin"], r["cout"], r["h"], r["w"], r["k"]], "dtype": r.get("dtype","")}

    r_pred = ranks(pred, higher_better=False)
    r_meas = ranks(meas, higher_better=False)

    out = {
        "n": len(meas),
        "spearman_pred_vs_meas": spearman(r_pred, r_meas),
        "mape_pred_vs_meas": mape(pred, meas),
        "pred": pred,
        "meas_p50_us": meas,
        "meta": meta,
        "weights": {"w_table": args.w_table, "w_cycles": args.w_cycles},
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print("spearman:", out["spearman_pred_vs_meas"], "mape:", out["mape_pred_vs_meas"])

if __name__ == "__main__":
    main()
