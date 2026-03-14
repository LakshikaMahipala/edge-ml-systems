from __future__ import annotations
import argparse
import pandas as pd
import numpy as np

def regret_at_k(df: pd.DataFrame, label_col: str, pred_col: str, k: int) -> float:
    df = df.dropna(subset=[label_col, pred_col])
    df = df.sort_values(pred_col, ascending=True)
    best_true = float(df[label_col].min())
    best_in_topk = float(df.head(k)[label_col].min())
    return best_in_topk - best_true

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked_csv", type=str, required=True)
    ap.add_argument("--label_col", type=str, default="y_fpga_est_total_ms")
    ap.add_argument("--pred_col", type=str, default="pred_latency_ms")
    ap.add_argument("--ks", type=str, default="5,10,20")
    args = ap.parse_args()

    df = pd.read_csv(args.ranked_csv)
    ks = [int(x) for x in args.ks.split(",")]

    print("Selection regret analysis")
    for k in ks:
        r = regret_at_k(df, args.label_col, args.pred_col, k)
        print(f"regret@{k}: {r:.6f} ms")

if __name__ == "__main__":
    main()
