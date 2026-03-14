from __future__ import annotations
import argparse
import numpy as np
import csv
from pathlib import Path

from utils import timer_ms
from conv2d_naive import conv2d_valid_single_channel
from winograd_f2x2_3x3 import winograd_tile_2x2

def winograd_conv_valid_single_channel(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Compute valid conv output using Winograd tiles.
    Output size is (H-2)x(W-2), assume that is divisible by 2 for simplicity.
    """
    H, W = d.shape
    OH, OW = H - 2, W - 2
    assert OH % 2 == 0 and OW % 2 == 0, "For simplicity require even output dims"
    out = np.zeros((OH, OW), dtype=np.float32)

    for oy in range(0, OH, 2):
        for ox in range(0, OW, 2):
            d4 = d[oy:oy+4, ox:ox+4]
            Y = winograd_tile_2x2(d4, g)
            out[oy:oy+2, ox:ox+2] = Y
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", type=str, default="results/winograd_results.csv")
    ap.add_argument("--H", type=int, default=34)  # gives (32x32) output
    ap.add_argument("--W", type=int, default=34)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    d = np.random.randn(args.H, args.W).astype(np.float32)
    g = np.random.randn(3, 3).astype(np.float32)

    ref = conv2d_valid_single_channel(d, g).astype(np.float32)
    wino = winograd_conv_valid_single_channel(d, g)

    max_err = float(np.max(np.abs(ref - wino)))

    r_naive = timer_ms(lambda: conv2d_valid_single_channel(d, g), iters=args.iters, warmup=args.warmup)
    r_wino  = timer_ms(lambda: winograd_conv_valid_single_channel(d, g), iters=args.iters, warmup=args.warmup)

    with out_p.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["H", "W", "method", "p50_ms", "p99_ms", "mean_ms", "max_abs_error"])
        w.writerow([args.H, args.W, "naive", r_naive["p50_ms"], r_naive["p99_ms"], r_naive["mean_ms"], max_err])
        w.writerow([args.H, args.W, "winograd_f2x2_3x3", r_wino["p50_ms"], r_wino["p99_ms"], r_wino["mean_ms"], max_err])

    print("Max abs error:", max_err)
    print("Wrote:", out_p)

if __name__ == "__main__":
    main()
