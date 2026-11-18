"""Accuracy-vs-latency plot from results.csv.

Usage: python src/plot_results.py results/t4/results.csv results/t4/accuracy_vs_latency.png
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: plot_results.py <results.csv> <out.png>")
    df = pd.read_csv(sys.argv[1])
    df["label"] = df["backend"] + " " + df["precision"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for model, g in df.groupby("model"):
        ax1.plot(g["p50_ms_b1"], 100 * g["top1"], marker="o", label=model)
        for _, r in g.iterrows():
            ax1.annotate(r["label"], (r["p50_ms_b1"], 100 * r["top1"]), fontsize=7,
                         xytext=(4, 2), textcoords="offset points")
    ax1.set(xlabel="Batch-1 latency, p50 (ms)", ylabel="Top-1 on Imagenette val (%)",
            title="Accuracy vs latency")
    ax1.grid(alpha=0.3)
    ax1.legend()

    pivot = df.pivot(index="label", columns="model", values="throughput_img_s_b32")
    pivot.plot.barh(ax=ax2)
    ax2.set(xlabel="Throughput at batch 32 (images/s)", ylabel="", title="Throughput")
    ax2.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)
    print(f"saved {sys.argv[2]}")


if __name__ == "__main__":
    main()
