# scripts/plot_results.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def load_json_files(results_dir: Path) -> List[Dict[str, Any]]:
    items = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="inference_bench/results")
    ap.add_argument("--out_dir", type=str, default="docs/plots")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_files(results_dir)
    if not data:
        print(f"No JSON files found in: {results_dir}")
        return

    labels = []
    p50s = []
    p99s = []

    for d in data:
        meta = d.get("meta", {})
        perf = d.get("perf", {})
        e2e = perf.get("end_to_end", {})

        model = meta.get("model", "unknown")
        device = meta.get("device", "unknown")
        inp = meta.get("input_desc", "unknown")

        p50 = e2e.get("p50_ms", None)
        p99 = e2e.get("p99_ms", None)
        if p50 is None or p99 is None:
            continue

        labels.append(f"{model}|{device}|{inp}")
        p50s.append(float(p50))
        p99s.append(float(p99))

    if not labels:
        print("No usable end-to-end p50/p99 found in JSON files.")
        return

    # Plot p50 and p99 as points (index-based)
    plt.figure()
    plt.plot(range(len(p50s)), p50s, marker="o")
    plt.plot(range(len(p99s)), p99s, marker="o")
    plt.xlabel("Run index")
    plt.ylabel("Latency (ms)")
    plt.title("End-to-end Latency: p50 vs p99")
    plt.legend(["p50", "p99"])
    out_path = out_dir / "e2e_p50_p99.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Plot p99/p50 ratio (tail amplification)
    ratios = [p99 / max(p50, 1e-9) for p50, p99 in zip(p50s, p99s)]
    plt.figure()
    plt.plot(range(len(ratios)), ratios, marker="o")
    plt.xlabel("Run index")
    plt.ylabel("p99 / p50")
    plt.title("Tail Amplification (p99/p50)")
    out_path2 = out_dir / "tail_amplification.png"
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path2}")

    # Save labels to a text file for mapping run index -> config
    label_path = out_dir / "run_labels.txt"
    label_path.write_text("\n".join([f"{i}: {lab}" for i, lab in enumerate(labels)]), encoding="utf-8")
    print(f"Saved: {label_path}")


if __name__ == "__main__":
    main()
