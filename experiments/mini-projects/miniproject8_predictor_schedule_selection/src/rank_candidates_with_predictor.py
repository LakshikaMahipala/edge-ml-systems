from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates_csv", type=str, required=True)
    ap.add_argument("--pred_col", type=str, default="pred_latency_ms")
    ap.add_argument("--out_csv", type=str, default="results/ranked.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.candidates_csv)

    # Placeholder: until model is trained, we rank by y_fpga_est_total_ms if present,
    # otherwise by bytes_total (proxy).
    if "y_fpga_est_total_ms" in df.columns:
        df[args.pred_col] = df["y_fpga_est_total_ms"]
    elif "bytes_total" in df.columns:
        df[args.pred_col] = df["bytes_total"]
    else:
        df[args.pred_col] = np.arange(len(df), dtype=np.float32)

    df = df.sort_values(args.pred_col, ascending=True)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    print("Wrote ranked candidates to:", out_p)

if __name__ == "__main__":
    main()
