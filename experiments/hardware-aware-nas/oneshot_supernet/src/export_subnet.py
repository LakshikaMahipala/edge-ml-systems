from __future__ import annotations
import argparse
import json
from utils import write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", type=str, default="results/best_subnet.json")
    ap.add_argument("--out", type=str, default="results/exported_subnet.json")
    args = ap.parse_args()

    best = json.loads(open(args.best, "r", encoding="utf-8").read())
    write_json(args.out, {"subnet": best["cfg"], "val_acc_sharedweights": best["acc"]})
    print("wrote:", args.out)

if __name__ == "__main__":
    main()
