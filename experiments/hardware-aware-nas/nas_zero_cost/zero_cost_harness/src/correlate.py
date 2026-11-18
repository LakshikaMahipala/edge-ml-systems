from __future__ import annotations
import argparse
import json
from typing import Dict, List, Tuple
from pathlib import Path

def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def ranks_from_scores(items: Dict[str, float], higher_better: bool = True) -> Dict[str, float]:
    # rank 1 = best
    sorted_ids = sorted(items.keys(), key=lambda k: items[k], reverse=higher_better)
    return {k: float(i + 1) for i, k in enumerate(sorted_ids)}

def spearman(r1: Dict[str, float], r2: Dict[str, float]) -> float:
    # only intersection
    keys = sorted(set(r1.keys()) & set(r2.keys()))
    n = len(keys)
    if n < 2:
        return 0.0
    d2 = 0.0
    for k in keys:
        d = r1[k] - r2[k]
        d2 += d * d
    return 1.0 - (6.0 * d2) / (n * (n*n - 1.0))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero_cost", type=str, default="results/zero_cost.jsonl")
    ap.add_argument("--proxy_scores", type=str, default="../nas_proxies/reduced_training_harness/results/proxy_scores.jsonl")
    ap.add_argument("--out", type=str, default="results/correlation.json")
    args = ap.parse_args()

    z = load_jsonl(args.zero_cost)
    p = load_jsonl(args.proxy_scores)

    # reduced-training entries only
    p = [r for r in p if r.get("type") == "proxy"]

    z_snip = {r["arch_id"]: float(r["snip_score"]) for r in z}
    z_gn   = {r["arch_id"]: float(r["gradnorm_score"]) for r in z}
    p_sc   = {r["arch_id"]: float(r["score"]) for r in p}

    r_snip = ranks_from_scores(z_snip, higher_better=True)
    r_gn   = ranks_from_scores(z_gn, higher_better=True)
    r_p    = ranks_from_scores(p_sc, higher_better=True)

    out = {
        "n_common_snip": len(set(r_snip) & set(r_p)),
        "n_common_gradnorm": len(set(r_gn) & set(r_p)),
        "spearman_snip_vs_reducedtrain": spearman(r_snip, r_p),
        "spearman_gradnorm_vs_reducedtrain": spearman(r_gn, r_p),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print(out)

if __name__ == "__main__":
    main()
