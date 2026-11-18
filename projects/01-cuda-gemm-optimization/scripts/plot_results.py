"""Plot GEMM benchmark results.

Usage:
    python scripts/plot_results.py results/gemm_t4.csv results/gemm_t4.png

Left panel: GFLOP/s vs matrix size for each kernel.
Right panel: each kernel's speed as a percentage of cuBLAS.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: plot_results.py <results.csv> <out.png>")
    df = pd.read_csv(sys.argv[1], comment="#")
    out = Path(sys.argv[2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for kernel, g in df.groupby("kernel", sort=False):
        ax1.plot(g["M"], g["gflops"], marker="o", label=kernel)
        if kernel != "cublas":
            ax2.plot(g["M"], g["pct_of_cublas"], marker="o", label=kernel)

    ax1.set(xscale="log", xlabel="Matrix size (M = N = K)", ylabel="GFLOP/s",
            title="SGEMM throughput")
    ax2.set(xscale="log", xlabel="Matrix size (M = N = K)", ylabel="% of cuBLAS",
            title="Speed relative to cuBLAS", ylim=(0, 105))
    for ax in (ax1, ax2):
        ax.set_xticks(sorted(df["M"].unique()))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
