from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import csv

from utils import timer_ms, pad_to_pow2
from strassen_numpy import strassen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="results/strassen_results.csv")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = ap.parse_args()

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    Ns = [64, 128, 256, 512, 1024]
    leaf_sizes = [32, 64, 128]

    dtype = np.float32 if args.dtype == "float32" else np.float64

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["N", "method", "leaf_size", "p50_ms", "p99_ms", "mean_ms", "max_abs_error"])

        for N in Ns:
            A = np.random.randn(N, N).astype(dtype)
            B = np.random.randn(N, N).astype(dtype)

            # reference
            C_ref = A @ B

            # numpy baseline
            r = timer_ms(lambda: A @ B, iters=args.iters, warmup=args.warmup)
            err = float(np.max(np.abs((A @ B) - C_ref)))
            w.writerow([N, "numpy_dot", "", r["p50_ms"], r["p99_ms"], r["mean_ms"], err])

            # Strassen (pad to pow2)
            Ap, n0 = pad_to_pow2(A)
            Bp, _ = pad_to_pow2(B)

            for leaf in leaf_sizes:
                def run_str():
                    Cp = strassen(Ap, Bp, leaf_size=leaf)
                    return Cp[:n0, :n0]

                r = timer_ms(run_str, iters=args.iters, warmup=args.warmup)
                C = run_str()
                err = float(np.max(np.abs(C - C_ref)))
                w.writerow([N, "strassen", leaf, r["p50_ms"], r["p99_ms"], r["mean_ms"], err])

    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
