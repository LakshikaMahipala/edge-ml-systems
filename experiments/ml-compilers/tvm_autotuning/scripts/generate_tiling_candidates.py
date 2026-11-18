from __future__ import annotations
import argparse
import json
from itertools import product
from pathlib import Path

def valid(cfg: dict, Cout: int, Cin: int) -> bool:
    tile_co = cfg["tile_co"]
    tile_ci = cfg["tile_ci"]
    vec = cfg["vec"]
    if vec > tile_co:
        return False
    if tile_co % vec != 0:
        return False
    if cfg["tile_y"] * cfg["tile_x"] < 4:
        return False
    # allow remainder tiles (so no strict divisibility required)
    if tile_co <= 0 or tile_ci <= 0:
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Cout", type=int, default=64)
    ap.add_argument("--Cin", type=int, default=64)
    ap.add_argument("--out", type=str, default="results/tiling_candidates.jsonl")
    args = ap.parse_args()

    knobs = {
        "tile_co": [4, 8, 16, 32, 64],
        "tile_ci": [4, 8, 16, 32, 64],
        "tile_y":  [1, 2, 4, 7, 14],
        "tile_x":  [1, 2, 4, 7, 14],
        "vec":     [1, 4, 8, 16],
        "unroll":  [0, 1],
    }

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    keys = list(knobs.keys())
    total = 0
    kept = 0

    with out_p.open("w", encoding="utf-8") as f:
        for values in product(*[knobs[k] for k in keys]):
            total += 1
            cfg = dict(zip(keys, values))
            if not valid(cfg, args.Cout, args.Cin):
                continue
            kept += 1
            cfg["shape"] = {"Cout": args.Cout, "Cin": args.Cin}
            f.write(json.dumps(cfg) + "\n")

    print(f"Wrote {kept} candidates to {out_p} (from {total} raw combos)")

if __name__ == "__main__":
    main()
