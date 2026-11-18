from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from search_space import default_search_space, sample_arch
from arch_encoding import arch_id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/candidates.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    space = default_search_space()

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            arch = sample_arch(space, rng=rng)
            rec = {"arch_id": arch_id(arch), "arch": arch}
            f.write(json.dumps(rec) + "\n")

    print("wrote:", out_p)

if __name__ == "__main__":
    main()
