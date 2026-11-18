from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim

from model import TinyMLP
from data_synth import make_synth_classification
from utils import train_epoch, eval_model, wall_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", type=str, default="results/qat_metrics.json")
    args = ap.parse_args()

    X, y = make_synth_classification()
    Xtr, ytr = X[:16000], y[:16000]
    Xva, yva = X[16000:], y[16000:]

    # QAT requires model in train mode and prepared with qconfig
    model = TinyMLP()
    model.train()

    # Eager-mode quantization setup
    # fbgemm is CPU server backend; qnnpack is mobile. Either is fine for demo.
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig("fbgemm")

    # Insert observers + fake-quant modules
    torch.ao.quantization.prepare_qat(model, inplace=True)

    opt = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(args.epochs):
        loss = train_epoch(model, opt, Xtr, ytr)
        acc = eval_model(model, Xva, yva)
        print(f"[QAT] epoch {e+1:02d}: loss={loss:.4f} val_acc={acc:.4f}")

    # Convert to a quantized model (int8 ops) for evaluation
    model.eval()
    qmodel = torch.ao.quantization.convert(model, inplace=False)

    # eval accuracy
    acc_q = eval_model(qmodel, Xva, yva)

    # timing
    x1 = Xva[:1]
    t = wall_ms(lambda: qmodel(x1))

    metrics = {"mode": "qat_int8", "val_acc": float(acc_q), "p50_ms_est": float(t)}
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(metrics, indent=2))
    print("wrote:", out_p)

if __name__ == "__main__":
    main()
