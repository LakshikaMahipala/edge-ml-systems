from __future__ import annotations
import argparse
import json
from pathlib import Path

from search_space import joint_space
from objective import acc_proxy, latency_proxy, energy_proxy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_all", type=str, default="results/codesign_points.jsonl")
    args = ap.parse_args()

    pts = []
    for i, cand in enumerate(joint_space()):
        arch = cand["arch"]
        hw = cand["hw"]

        acc = acc_proxy(arch)
        lat = latency_proxy(arch, hw)
        ene = energy_proxy(arch, hw)

        rec = {
            "id": f"P{i:04d}",
            "arch": arch,
            "hw": hw,
            "acc": float(acc),
            "latency": float(lat),
            "energy": float(ene),
        }
        pts.append(rec)

    out_p = Path(args.out_all)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        for r in pts:
            f.write(json.dumps(r) + "\n")

    print("wrote:", out_p, "n=", len(pts))

if __name__ == "__main__":
    main()
