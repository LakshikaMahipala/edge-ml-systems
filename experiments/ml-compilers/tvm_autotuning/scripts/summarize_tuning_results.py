from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=str, required=True)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    print("Tuning summary")
    for k, v in meta.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
