from __future__ import annotations
import argparse
from typing import List, Dict, Any

from candidate_loader import load_candidates
from tiny_train_proxy import train_proxy_once
from aging_scheduler import AgingConfig
from log_utils import append_jsonl, write_json

def leaderboard_top(scores: List[Dict[str, Any]], top_m: int):
    return sorted(scores, key=lambda r: r["score"], reverse=True)[:top_m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="../nas_foundations/tiny_cnn_search_space/results/candidates.jsonl")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=str, default="results/proxy_scores.jsonl")
    ap.add_argument("--rerank_out", type=str, default="results/rerank.json")
    ap.add_argument("--every_k", type=int, default=20)
    ap.add_argument("--top_m", type=int, default=5)
    ap.add_argument("--extra_steps", type=int, default=60)
    args = ap.parse_args()

    aging = AgingConfig(every_k=args.every_k, top_m=args.top_m, extra_steps=args.extra_steps)

    all_scores: List[Dict[str, Any]] = []
    snapshots: List[Dict[str, Any]] = []

    for idx, rec in enumerate(load_candidates(args.candidates)):
        arch_id = rec["arch_id"]
        arch = rec["arch"]

        ps = train_proxy_once(arch_id, arch, steps=args.steps, seed=args.seed + idx)
        d = ps.to_dict()
        all_scores.append(d)
        append_jsonl(args.log, {"type":"proxy", "idx": idx, **d})

        # aging step
        if (idx + 1) % aging.every_k == 0:
            before = leaderboard_top(all_scores, aging.top_m)

            refreshed = []
            for item in before:
                # re-evaluate with more steps + new seed
                ps2 = train_proxy_once(item["arch_id"], rec_from_id(all_scores, item["arch_id"])["arch"],
                                       steps=aging.extra_steps, seed=args.seed + 10000 + idx)
                dd = ps2.to_dict()
                refreshed.append(dd)
                append_jsonl(args.log, {"type":"aging_refresh", "at_idx": idx, **dd})

                # update stored score (replace best-known)
                for k in range(len(all_scores)):
                    if all_scores[k]["arch_id"] == item["arch_id"]:
                        # keep higher score as "best known" proxy
                        if dd["score"] > all_scores[k]["score"]:
                            all_scores[k] = {**all_scores[k], **dd}
                        break

            after = leaderboard_top(all_scores, aging.top_m)
            snapshots.append({"at_candidate": idx, "before": before, "after": after})

    write_json(args.rerank_out, {"snapshots": snapshots})
    print("log:", args.log)
    print("rerank:", args.rerank_out)

def rec_from_id(all_scores: List[Dict[str, Any]], arch_id: str) -> Dict[str, Any]:
    # in this harness, we need arch to re-eval; easiest: store arch alongside scores during first pass
    for r in all_scores:
        if r["arch_id"] == arch_id and "arch" in r:
            return r
    raise KeyError(f"arch not found for id={arch_id}")

if __name__ == "__main__":
    main()
