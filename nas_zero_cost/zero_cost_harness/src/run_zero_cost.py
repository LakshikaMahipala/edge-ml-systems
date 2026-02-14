from __future__ import annotations
import argparse
import random

import torch

from candidate_loader import load_candidates
from data_onebatch import make_one_batch
from snip_proxy import snip_like_score
from gradnorm_proxy import gradnorm_score
from log_utils import append_jsonl

# run later with:
# PYTHONPATH=../nas_foundations/tiny_cnn_search_space/src python src/run_zero_cost.py ...
from model_builder import build_model
from proxy_metrics import count_params, macs_proxy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=str, default="../nas_foundations/tiny_cnn_search_space/results/candidates.jsonl")
    ap.add_argument("--log", type=str, default="results/zero_cost.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max", type=int, default=100)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    for idx, rec in enumerate(load_candidates(args.candidates)):
        if idx >= args.max:
            break
        arch_id = rec["arch_id"]
        arch = rec["arch"]

        model = build_model(arch)
        params = count_params(model)
        macs = macs_proxy(model)

        X, y = make_one_batch(batch=args.batch, num_classes=arch["num_classes"], seed=args.seed + idx)

        snip_s, snip_c = snip_like_score(model, X, y)
        # rebuild to avoid any state/grad leftovers affecting second proxy
        model2 = build_model(arch)
        gn_s, gn_c = gradnorm_score(model2, X, y)

        out = {
            "arch_id": arch_id,
            "idx": idx,
            "params": int(params),
            "macs": int(macs),
            "snip_score": float(snip_s),
            "gradnorm_score": float(gn_s),
            "snip_components": snip_c,
            "gradnorm_components": gn_c,
        }
        append_jsonl(args.log, out)

        if (idx + 1) % 20 == 0:
            print("scored", idx + 1)

    print("done:", args.log)

if __name__ == "__main__":
    main()
