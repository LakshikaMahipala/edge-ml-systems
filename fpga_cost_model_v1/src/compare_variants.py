from __future__ import annotations
import argparse
import csv
from pathlib import Path

from cost_model_v1 import estimate_point

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="results/compare_lowrank_vs_base.csv")
    ap.add_argument("--IN", type=int, default=1024)
    ap.add_argument("--OUT", type=int, default=1024)
    ap.add_argument("--ranks", type=str, default="64,128,256,512")
    ap.add_argument("--unroll", type=int, default=16)
    ap.add_argument("--II", type=int, default=1)
    ap.add_argument("--baud", type=float, default=115200)
    ap.add_argument("--f_clk_mhz", type=float, default=100)
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    base = estimate_point({
        "variant": "BASE_INT8_FC",
        "IN": args.IN, "OUT": args.OUT,
        "UNROLL": args.unroll, "II": args.II,
        "baud": args.baud, "f_clk_mhz": args.f_clk_mhz
    })

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["variant","IN","OUT","rank_r","UNROLL","II",
                    "bytes_total","macs","dsp","bram",
                    "io_ms","compute_ms","transform_ms","total_ms"])

        w.writerow([
            base["variant"], args.IN, args.OUT, "",
            args.unroll, args.II,
            base["bytes_total"], base["macs"],
            base["resource_proxy_dsp"], base["resource_proxy_bram"],
            base["y_fpga_est_io_ms"], base["y_fpga_est_compute_ms"], base["y_fpga_est_transform_ms"],
            base["y_fpga_est_total_ms"]
        ])

        for r in ranks:
            lr = estimate_point({
                "variant": "LOWRANK_INT8_FC",
                "IN": args.IN, "OUT": args.OUT,
                "rank_r": r,
                "UNROLL": args.unroll, "II": args.II,
                "baud": args.baud, "f_clk_mhz": args.f_clk_mhz
            })
            w.writerow([
                lr["variant"], args.IN, args.OUT, r,
                args.unroll, args.II,
                lr["bytes_total"], lr["macs"],
                lr["resource_proxy_dsp"], lr["resource_proxy_bram"],
                lr["y_fpga_est_io_ms"], lr["y_fpga_est_compute_ms"], lr["y_fpga_est_transform_ms"],
                lr["y_fpga_est_total_ms"]
            ])

    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
