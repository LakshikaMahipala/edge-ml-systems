from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from hashlib import blake2b

def make_id(d: dict) -> str:
    h = blake2b(digest_size=8)
    h.update(json.dumps(d, sort_keys=True).encode("utf-8"))
    return h.hexdigest()

def gen_fc(rng: random.Random) -> dict:
    IN = rng.choice([32, 64, 128, 256, 512, 1024])
    OUT = rng.choice([16, 32, 64, 128, 256])
    SHIFT = rng.choice([6, 7, 8])
    d = {"op_type": "INT8_FC", "IN": IN, "OUT": OUT, "SHIFT": SHIFT}
    d["config_id"] = make_id(d)
    return d

def gen_dwconv1d(rng: random.Random) -> dict:
    C = rng.choice([4, 8, 16, 32, 64])
    L = rng.choice([16, 32, 64, 128, 256])
    K = 3
    SHIFT = rng.choice([6, 7, 8])
    d = {"op_type": "INT8_DWCONV1D_K3", "C": C, "L": L, "K": K, "SHIFT": SHIFT}
    d["config_id"] = make_id(d)
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/configs.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            if rng.random() < 0.5:
                row = gen_fc(rng)
            else:
                row = gen_dwconv1d(rng)
            row["seed"] = args.seed
            f.write(json.dumps(row) + "\n")

    print("Wrote configs to:", out_p)

if __name__ == "__main__":
    main()
