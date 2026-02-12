from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from model import TinyMLP
from data_synth import make_synth_classification
from utils import eval_model, wall_ms

def prune_hidden_neurons(model: TinyMLP, keep_ratio: float) -> TinyMLP:
    """
    Structured pruning: reduce hidden dimension by selecting neurons with largest L2 norm
    in fc1 and fc2. Produces a NEW smaller dense model.
    """
    assert 0 < keep_ratio <= 1.0
    h = model.fc1.out_features
    h_keep = max(1, int(h * keep_ratio))

    with torch.no_grad():
        # score neurons by L2 norm of incoming weights
        s1 = torch.norm(model.fc1.weight, p=2, dim=1)  # (h,)
        idx1 = torch.topk(s1, k=h_keep, largest=True).indices.sort().values

        s2 = torch.norm(model.fc2.weight, p=2, dim=1)  # (h,)
        idx2 = torch.topk(s2, k=h_keep, largest=True).indices.sort().values

        # build new model
        new = TinyMLP(in_dim=model.fc1.in_features, hidden=h_keep, num_classes=model.fc3.out_features)

        # fc1: select rows
        new.fc1.weight.copy_(model.fc1.weight[idx1])
        new.fc1.bias.copy_(model.fc1.bias[idx1])

        # fc2: select submatrix
        # new.fc2.weight shape (h_keep, h_keep): take rows idx2 and cols idx1
        new.fc2.weight.copy_(model.fc2.weight[idx2][:, idx1])
        new.fc2.bias.copy_(model.fc2.bias[idx2])

        # fc3: input dim reduced, select cols idx2
        new.fc3.weight.copy_(model.fc3.weight[:, idx2])
        new.fc3.bias.copy_(model.fc3.bias)

    return new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_ckpt", type=str, default="results/base_model.pt")
    ap.add_argument("--keep_ratio", type=float, default=0.5)
    ap.add_argument("--out", type=str, default="results/pruned_structured_metrics.json")
    args = ap.parse_args()

    X, y = make_synth_classification()
    Xva, yva = X[16000:], y[16000:]

    base = TinyMLP()
    base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    base.eval()

    acc_before = eval_model(base, Xva, yva)
    t_before = wall_ms(lambda: base(Xva[:1]))

    pruned = prune_hidden_neurons(base, args.keep_ratio)
    pruned.eval()

    acc_after = eval_model(pruned, Xva, yva)
    t_after = wall_ms(lambda: pruned(Xva[:1]))

    # param count compare
    p_before = sum(p.numel() for p in base.parameters())
    p_after = sum(p.numel() for p in pruned.parameters())

    metrics = {
        "mode": "structured_prune",
        "keep_ratio": float(args.keep_ratio),
        "val_acc_before": float(acc_before),
        "val_acc_after": float(acc_after),
        "p50_ms_est_before": float(t_before),
        "p50_ms_est_after": float(t_after),
        "params_before": int(p_before),
        "params_after": int(p_after),
        "param_reduction": float(1.0 - (p_after / p_before)),
        "note": "Structured pruning changes shapes so dense kernels can speed up.",
    }

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(metrics, indent=2))
    print("wrote:", out_p)

if __name__ == "__main__":
    main()
