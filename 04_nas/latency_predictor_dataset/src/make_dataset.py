from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

COLUMNS = [
    "config_id","op_type","IN","OUT","C","L","K","SHIFT",
    "bytes_in","bytes_out","bytes_total","macs","cycles_est",
    "interface","baud","f_clk_mhz",
    "y_fpga_est_io_ms","y_fpga_est_compute_ms","y_fpga_est_total_ms",
    "y_cpu_measured_ms","y_gpu_measured_ms",
    "seed","notes"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="data/labeled.jsonl")
    ap.add_argument("--out_csv", type=str, default="data/dataset.csv")
    args = ap.parse_args()

    in_p = Path(args.in_jsonl)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with in_p.open("r", encoding="utf-8") as fi, out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=COLUMNS)
        w.writeheader()
        for line in fi:
            row = json.loads(line)
            # fill missing columns
            for c in COLUMNS:
                row.setdefault(c, "")
            w.writerow({c: row.get(c, "") for c in COLUMNS})

    print("Wrote CSV dataset:", out_p)

if __name__ == "__main__":
    main()
