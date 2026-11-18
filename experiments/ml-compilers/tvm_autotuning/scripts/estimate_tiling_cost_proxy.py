from __future__ import annotations
import argparse
import json
from pathlib import Path

def approx_conv_macs(N: int, Cout: int, H: int, W: int, Cin: int, KH: int, KW: int) -> int:
    return N * Cout * H * W * Cin * KH * KW

def approx_bytes(Cout: int, Cin: int, H: int, W: int, KH: int, KW: int) -> int:
    # very rough: bytes for input + weights + output in int8
    x = Cin * H * W
    w = Cout * Cin * KH * KW
    y = Cout * H * W
    return x + w + y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="results/tiling_candidates.jsonl")
    ap.add_argument("--out_csv", type=str, default="results/tiling_proxy_scores.csv")
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--H", type=int, default=56)
    ap.add_argument("--W", type=int, default=56)
    ap.add_argument("--Cout", type=int, default=64)
    ap.add_argument("--Cin", type=int, default=64)
    ap.add_argument("--KH", type=int, default=3)
    ap.add_argument("--KW", type=int, default=3)
    args = ap.parse_args()

    cand_p = Path(args.candidates)
    out_p = Path(args.out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    macs = approx_conv_macs(args.N, args.Cout, args.H, args.W, args.Cin, args.KH, args.KW)
    base_bytes = approx_bytes(args.Cout, args.Cin, args.H, args.W, args.KH, args.KW)

    # Proxy: favor larger tiles (more reuse) but penalize huge tiles (cache pressure)
    # This is a teaching heuristic, not physics.
    def proxy(cfg: dict) -> float:
        reuse = cfg["tile_y"] * cfg["tile_x"] * cfg["tile_ci"]
        cache_penalty = (cfg["tile_co"] * cfg["tile_ci"]) / 256.0
        return (base_bytes / max(reuse, 1)) * (1.0 + cache_penalty)

    rows = []
    with cand_p.open("r", encoding="utf-8") as f:
        for line in f:
            cfg = json.loads(line)
            s = proxy(cfg)
            rows.append((s, cfg))

    rows.sort(key=lambda x: x[0])

    with out_p.open("w", encoding="utf-8") as f:
        f.write("proxy_score,tile_co,tile_ci,tile_y,tile_x,vec,unroll\n")
        for s, cfg in rows[:200]:  # write top 200 to keep file small
            f.write(f"{s:.6f},{cfg['tile_co']},{cfg['tile_ci']},{cfg['tile_y']},{cfg['tile_x']},{cfg['vec']},{cfg['unroll']}\n")

    print("MACs (fixed):", macs)
    print("Base bytes (approx):", base_bytes)
    print("Wrote ranked proxy scores to:", out_p)
    print("Top-5 configs:")
    for s, cfg in rows[:5]:
        print(s, cfg)

if __name__ == "__main__":
    main()
