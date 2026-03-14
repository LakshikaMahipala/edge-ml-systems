from __future__ import annotations
import argparse
from pathlib import Path
import json

import pandas as pd
import numpy as np

from featurize import make_xy
from split import group_split
from evaluate import report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_csv", type=str, default="../latency_predictor_dataset/data/dataset.csv")
    ap.add_argument("--target", type=str, default="y_fpga_est_total_ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="results/sklearn_baseline")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor

    df = pd.read_csv(args.dataset_csv)
    train_df, val_df, test_df = group_split(df, seed=args.seed)

    Xtr, ytr, ytr_log, cols = make_xy(train_df, args.target)
    Xva, yva, yva_log, _ = make_xy(val_df, args.target)
    Xte, yte, yte_log, _ = make_xy(test_df, args.target)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=args.seed
    )

    # train on log-latency
    model.fit(Xtr, ytr_log)

    # predict then invert transform
    def inv(z): return np.expm1(np.clip(z, -50, 50))
    yva_hat = inv(model.predict(Xva))
    yte_hat = inv(model.predict(Xte))

    val_metrics = report(yva, yva_hat)
    test_metrics = report(yte, yte_hat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps({
        "val": val_metrics,
        "test": test_metrics,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "target": args.target,
        "seed": args.seed
    }, indent=2), encoding="utf-8")

    (out_dir / "feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")

    print("Saved metrics to:", out_dir / "metrics.json")
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)

if __name__ == "__main__":
    main()
