from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from fp32_model import FP32MLP
from data_synth import make_synth_classification
from utils import train_epoch, eval_model, wall_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/fp32_metrics.json")
    args = ap.parse_args()

    X, y = make_synth_classification()
    Xtr, ytr = X[:16000], y[:16000]
    Xva, yva = X[16000:], y[16000:]

    model = FP32MLP()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(args.epochs):
        loss = train_epoch(model, opt, Xtr, ytr)
        acc = eval_model(model, Xva, yva)
        print(f"[FP32] epoch {e+1:02d}: loss={loss:.4f} val_acc={acc:.4f}")

    model.eval()
    t = wall_ms(lambda: model(Xva[:1]))
    metrics = {"mode": "fp32", "val_acc": float(eval_model(model, Xva, yva)), "p50_ms_est": float(t)}

    out_p = Path(args.out); out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(metrics, indent=2))
    print("wrote:", out_p)

if __name__ == "__main__":
    main()
