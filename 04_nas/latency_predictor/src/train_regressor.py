from __future__ import annotations
import argparse
import json
import torch
import torch.optim as optim

from dataset import load_jsonl, train_val_split
from features import graph_to_features, features_to_vector
from model_regressor import MLPRegressor
from eval_metrics import spearman, mape, rmse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/regression_results.json")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_jsonl(args.data)
    tr, va = train_val_split(rows, val_ratio=0.2, seed=args.seed)

    feats_tr = [graph_to_features(r) for r in tr]
    keys = sorted(feats_tr[0].keys())
    Xtr = torch.tensor([features_to_vector(f, keys) for f in feats_tr], dtype=torch.float32)
    ytr = torch.tensor([r["target"]["latency_p50_ms"] for r in tr], dtype=torch.float32)

    feats_va = [graph_to_features(r) for r in va]
    Xva = torch.tensor([features_to_vector(f, keys) for f in feats_va], dtype=torch.float32)
    yva = torch.tensor([r["target"]["latency_p50_ms"] for r in va], dtype=torch.float32)

    torch.manual_seed(args.seed)
    model = MLPRegressor(d_in=Xtr.shape[1])
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(args.epochs):
        model.train()
        pred = model(Xtr)
        loss = torch.nn.functional.mse_loss(pred, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pva = model(Xva).cpu().numpy().tolist()
        tva = yva.cpu().numpy().tolist()

    out = {
        "n_train": len(tr),
        "n_val": len(va),
        "spearman": spearman(tva, pva),
        "mape": mape(tva, pva),
        "rmse": rmse(tva, pva),
        "feature_keys": keys,
        "note": "baseline MLP regressor; replace with GNN later",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("wrote:", args.out)
    print(out)

if __name__ == "__main__":
    main()
