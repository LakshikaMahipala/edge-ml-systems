from __future__ import annotations
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from utils import timer_ms
from compress_linear_torch import compress_linear_svd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dim", type=int, default=1024)
    ap.add_argument("--out_dim", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ranks", type=str, default="64,128,256,512")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out_csv", type=str, default="results/low_rank_results.csv")
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    x = torch.randn(args.batch, args.in_dim)

    base = nn.Linear(args.in_dim, args.out_dim, bias=True)
    base.eval()

    with torch.no_grad():
        y_ref = base(x)

    def run_base():
        with torch.no_grad():
            _ = base(x)

    r_base = timer_ms(run_base, iters=args.iters, warmup=args.warmup)

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["in_dim","out_dim","batch","rank","method","p50_ms","p99_ms","mean_ms","max_abs_err","rel_err"])

        w.writerow([args.in_dim,args.out_dim,args.batch,"", "baseline",
                    r_base["p50_ms"], r_base["p99_ms"], r_base["mean_ms"], 0.0, 0.0])

        for r in ranks:
            lr = compress_linear_svd(base, r)
            lr.eval()

            with torch.no_grad():
                y = lr(x)
            err = float(torch.max(torch.abs(y - y_ref)).item())
            rel = float(err / (torch.max(torch.abs(y_ref)).item() + 1e-9))

            def run_lr():
                with torch.no_grad():
                    _ = lr(x)

            rr = timer_ms(run_lr, iters=args.iters, warmup=args.warmup)
            w.writerow([args.in_dim,args.out_dim,args.batch,r,"low_rank",
                        rr["p50_ms"], rr["p99_ms"], rr["mean_ms"], err, rel])

    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
