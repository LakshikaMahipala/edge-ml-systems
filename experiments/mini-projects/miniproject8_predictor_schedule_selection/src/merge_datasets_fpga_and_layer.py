from __future__ import annotations
import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer_csv", type=str, default="../latency_predictor_dataset/data/dataset.csv")
    ap.add_argument("--fpga_csv", type=str, default="../fpga_tiling_sweep/data/fpga_sweep_dataset.csv")
    ap.add_argument("--out_csv", type=str, default="results/merged_dataset.csv")
    args = ap.parse_args()

    layer = pd.read_csv(args.layer_csv)
    fpga = pd.read_csv(args.fpga_csv)

    # tag backend
    layer["backend"] = "layer_est"   # generic layer estimated dataset
    fpga["backend"] = "fpga_sweep"

    # normalize column presence
    for col in ["tile_co","tile_ci","tile_y","tile_x","vec","unroll"]:
        if col not in layer.columns: layer[col] = ""
        if col not in fpga.columns: fpga[col] = ""

    out = pd.concat([layer, fpga], ignore_index=True, sort=False)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_p, index=False)
    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
