from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

from utils import timer_ms
from conv_naive_1d import conv1d_full_naive
from fft_conv_1d import conv1d_full_fft
from conv2d_naive import conv2d_valid_single_channel
from fft_conv_2d import conv2d_valid_fft

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="results/fft_conv_1d_crossover.csv")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    Ns = [256, 512, 1024, 2048, 4096, 8192]
    Ks = [3, 7, 15, 31, 63, 127, 255]

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["N", "K", "method", "p50_ms", "p99_ms", "mean_ms", "max_abs_error"])

        for N in Ns:
            for K in Ks:
                x = np.random.randn(N).astype(np.float32)
                h = np.random.randn(K).astype(np.float32)

                y_ref = conv1d_full_naive(x, h)
                y_fft = conv1d_full_fft(x, h, pad_pow2=True)
                err = float(np.max(np.abs(y_ref - y_fft)))

                r_naive = timer_ms(lambda: conv1d_full_naive(x, h), iters=args.iters, warmup=args.warmup)
                r_fft   = timer_ms(lambda: conv1d_full_fft(x, h, pad_pow2=True), iters=args.iters, warmup=args.warmup)

                w.writerow([N, K, "naive", r_naive["p50_ms"], r_naive["p99_ms"], r_naive["mean_ms"], err])
                w.writerow([N, K, "fft_pow2", r_fft["p50_ms"], r_fft["p99_ms"], r_fft["mean_ms"], err])

    # quick 2D correctness spot-check (not timed)
    d = np.random.randn(34, 34).astype(np.float32)
    g = np.random.randn(3, 3).astype(np.float32)
    ref2 = conv2d_valid_single_channel(d, g)
    fft2 = conv2d_valid_fft(d, g)
    print("2D spot-check max error:", float(np.max(np.abs(ref2 - fft2))))
    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
