from __future__ import annotations
import numpy as np
import pandas as pd

def group_split(df: pd.DataFrame, group_col: str = "config_id", seed: int = 0,
                train_frac: float = 0.7, val_frac: float = 0.15):
    rng = np.random.default_rng(seed)
    groups = df[group_col].astype(str).unique()
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_g = set(groups[:n_train])
    val_g = set(groups[n_train:n_train+n_val])
    test_g = set(groups[n_train+n_val:])

    train_df = df[df[group_col].astype(str).isin(train_g)].copy()
    val_df   = df[df[group_col].astype(str).isin(val_g)].copy()
    test_df  = df[df[group_col].astype(str).isin(test_g)].copy()

    return train_df, val_df, test_df
