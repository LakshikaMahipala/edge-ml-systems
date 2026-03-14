from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

def run(cmd):
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--amount", type=float, default=0.5)
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    args = ap.parse_args()

    run([sys.executable, "src/train_base.py", "--epochs", str(args.epochs)])
    run([sys.executable, "src/prune_unstructured.py", "--amount", str(args.amount)])
    run([sys.executable, "src/prune_structured.py", "--keep_ratio", str(args.keep_ratio)])

    # Summarize
    out = {}
    for p in ["results/base_metrics.json",
              "results/pruned_unstructured_metrics.json",
              "results/pruned_structured_metrics.json"]:
        pp = Path(p)
        if pp.exists():
            out[p] = json.loads(pp.read_text())

    Path("results/summary.json").write_text(json.dumps(out, indent=2))
    print("wrote: results/summary.json")

if __name__ == "__main__":
    main()
