from __future__ import annotations
import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from featurize import make_xy
from split import group_split
from evaluate import report

class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, default="../latency_predictor_dataset/data/dataset.csv")
    ap.add_argument("--target", type=str, default="y_fpga_est_total_ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", type=str, default="results/mlp_torch")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.dataset_csv)
    train_df, val_df, test_df = group_split(df, seed=args.seed)

    Xtr, ytr, ytr_log, cols = make_xy(train_df, args.target)
    Xva, yva, yva_log, _ = make_xy(val_df, args.target)
    Xte, yte, yte_log, _ = make_xy(test_df, args.target)

    # standardize features (important for MLP)
    mu = Xtr.mean(axis=0, keepdims=True)
    sig = Xtr.std(axis=0, keepdims=True) + 1e-6
    Xtrn = (Xtr - mu) / sig
    Xvan = (Xva - mu) / sig
    Xten = (Xte - mu) / sig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(Xtrn.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def batch_iter(X, y, bs=128):
        n = len(X)
        idx = np.random.permutation(n)
        for i in range(0, n, bs):
            j = idx[i:i+bs]
            yield X[j], y[j]

    best_val = float("inf")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in batch_iter(Xtrn, ytr_log):
            xb = torch.tensor(xb, dtype=torch.float32, device=device)
            yb = torch.tensor(yb, dtype=torch.float32, device=device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            xva = torch.tensor(Xvan, dtype=torch.float32, device=device)
            pred_va = model(xva).cpu().numpy()

        val_mse = float(np.mean((pred_va - yva_log) ** 2))
        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "state_dict": model.state_dict(),
                "mu": mu,
                "sig": sig,
                "cols": cols
            }, out_dir / "best.pt")

        if ep % 10 == 0 or ep == 1:
            print(f"epoch={ep:03d} train_mse~{np.mean(tr_losses):.6f} val_mse={val_mse:.6f}")

    # load best + final eval in ms
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def inv(z): return np.expm1(np.clip(z, -50, 50))

    with torch.no_grad():
        yva_hat = inv(model(torch.tensor(Xvan, dtype=torch.float32, device=device)).cpu().numpy())
        yte_hat = inv(model(torch.tensor(Xten, dtype=torch.float32, device=device)).cpu().numpy())

    metrics = {
        "val": report(yva, yva_hat),
        "test": report(yte, yte_hat),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "target": args.target,
        "seed": args.seed,
        "device": device
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out_dir / "metrics.json")
    print("VAL:", metrics["val"])
    print("TEST:", metrics["test"])

if __name__ == "__main__":
    main()
