import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt

# Usage:
# python scripts/plot_results.py results/summary.csv results/plot_gflops.png

def main():
    if len(sys.argv) != 3:
        print("Usage: plot_results.py <summary_csv> <out_png>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    out_png = Path(sys.argv[2])

    benches = []
    gflops = []

    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            b = row["bench"]
            val = row["gflops"]
            if val.strip() == "":
                continue
            benches.append(b)
            gflops.append(float(val))

    plt.figure()
    plt.bar(range(len(benches)), gflops)
    plt.xticks(range(len(benches)), benches, rotation=30, ha="right")
    plt.ylabel("GFLOP/s")
    plt.title("CUDA Microbench Throughput")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
