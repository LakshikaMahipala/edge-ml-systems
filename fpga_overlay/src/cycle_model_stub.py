from __future__ import annotations
import argparse
import math
import json
from pathlib import Path

def cycles_conv_tile(Tm, Tr, Tc, Tn, K, P, bw_bytes_per_cycle=16.0, overhead=200):
    ops = Tm * Tr * Tc * Tn * (K*K)
    cycles_compute = ops / max(P, 1)

    # crude memory bytes: input tile + weight tile + output tile
    bytes_in = Tn * (Tr+K-1) * (Tc+K-1)
    bytes_w  = Tm * Tn * (K*K)
    bytes_out = Tm * Tr * Tc
    bytes_total = bytes_in + bytes_w + bytes_out

    cycles_mem = bytes_total / max(bw_bytes_per_cycle, 1e-9)

    return float(max(cycles_compute, cycles_mem) + overhead)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Tm", type=int, default=8)
    ap.add_argument("--Tn", type=int, default=8)
    ap.add_argument("--Tr", type=int, default=8)
    ap.add_argument("--Tc", type=int, default=8)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--P", type=int, default=64)
    ap.add_argument("--out", type=str, default="figures/cycle_model_example.json")
    args = ap.parse_args()

    c = cycles_conv_tile(args.Tm,args.Tr,args.Tc,args.Tn,args.K,args.P)
    out = {"cycles_tile_proxy": c, "params": vars(args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out, out)

if __name__ == "__main__":
    main()
