from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from model import TinyMLP
from data_synth import make_synth_classification
from utils import train_epoch, eval_model, wall_ms, model_sparsity

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/base_metrics.json")
    ap.add_argument("--save_model", type=str, default="results/base_model.pt")
    args = ap.parse_args()

    X, y = make_synth_classification()
    Xtr, ytr = X[:16000], y[:16000]
    Xva, yva = X[16000:], y[16000:]

    model = TinyMLP()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(args.epochs):
        loss = train_epoch(model, opt, Xtr, ytr)
        acc = eval_model(model, Xva, yva)
        print(f"epoch {e+1:02d}: loss={loss:.4f} val_acc={acc:.4f}")

    model.eval()
    x1 = Xva[:1]
    t = wall_ms(lambda: model(x1))

    metrics = {
        "mode": "baseline",
        "val_acc": float(eval_model(model, Xva, yva)),
        "p50_ms_est": float(t),
        "model_sparsity": float(model_sparsity(model)),
    }

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(metrics, indent=2))
    torch.save(model.state_dict(), Path(args.save_model))

    print("wrote:", out_p)
    print("saved:", args.save_model)

if __name__ == "__main__":
    main()
