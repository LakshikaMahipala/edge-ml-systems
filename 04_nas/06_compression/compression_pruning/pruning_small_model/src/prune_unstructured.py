from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn.utils.prune as prune

from model import TinyMLP
from data_synth import make_synth_classification
from utils import eval_model, wall_ms, model_sparsity

def apply_global_unstructured(model: torch.nn.Module, amount: float):
    params = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            params.append((m, "weight"))
    prune.global_unstructured(
        params,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    # make pruning permanent: remove reparam
    for m, pname in params:
        prune.remove(m, pname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, default="results/base_model.pt")
    ap.add_argument("--amount", type=float, default=0.5)  # 50% weights pruned
    ap.add_argument("--out", type=str, default="results/pruned_unstructured_metrics.json")
    args = ap.parse_args()

    X, y = make_synth_classification()
    Xva, yva = X[16000:], y[16000:]

    model = TinyMLP()
    model.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    model.eval()

    acc_before = eval_model(model, Xva, yva)
    t_before = wall_ms(lambda: model(Xva[:1]))

    apply_global_unstructured(model, args.amount)
    model.eval()

    acc_after = eval_model(model, Xva, yva)
    t_after = wall_ms(lambda: model(Xva[:1]))
    sp = model_sparsity(model)

    metrics = {
        "mode": "unstructured_prune",
        "amount": float(args.amount),
        "val_acc_before": float(acc_before),
        "val_acc_after": float(acc_after),
        "p50_ms_est_before": float(t_before),
        "p50_ms_est_after": float(t_after),
        "model_sparsity": float(sp),
        "note": "Unstructured pruning may not speed up without sparse kernels.",
    }

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(metrics, indent=2))
    print("wrote:", out_p)

if __name__ == "__main__":
    main()
