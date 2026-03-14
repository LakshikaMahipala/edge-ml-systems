from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

from estimate_fpga_point_cost import estimate

COLUMNS = [
    "op_type","IN","OUT","C","L","K",
    "UNROLL","UNROLL_C","II",
    "baud","f_clk_mhz",
    "bytes_in","bytes_out","bytes_total",
    "macs","cycles_est","resource_proxy_dsp",
    "y_fpga_est_io_ms","y_fpga_est_compute_ms","y_fpga_est_total_ms",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="data/sweep_points.jsonl")
    ap.add_argument("--out_csv", type=str, default="data/fpga_sweep_dataset.csv")
    args = ap.parse_args()

    in_p = Path(args.in_jsonl)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with in_p.open("r", encoding="utf-8") as fi, out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLUMNS)
        w.writeheader()
        for line in fi:
            row = json.loads(line)
            labeled = estimate(row)
            # ensure all cols exist
            for c in COLUMNS:
                labeled.setdefault(c, "")
            w.writerow({c: labeled.get(c, "") for c in COLUMNS})

    print("Wrote FPGA sweep dataset to:", out_p)

if __name__ == "__main__":
    main()
