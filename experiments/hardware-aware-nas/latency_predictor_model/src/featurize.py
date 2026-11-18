from __future__ import annotations
import math
import pandas as pd
import numpy as np

CATEGORICALS = ["op_type", "interface"]
NUMERICALS = [
    "IN","OUT","C","L","K","SHIFT",
    "bytes_in","bytes_out","bytes_total",
    "macs","cycles_est",
    "baud","f_clk_mhz",
]

def _safe_num(x):
    try:
        if x == "" or pd.isna(x):
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # ensure numeric
    for c in NUMERICALS:
        if c in df.columns:
            df[c] = df[c].map(_safe_num)

    # arithmetic intensity: macs / bytes_total
    df["arith_intensity"] = df["macs"] / (df["bytes_total"] + 1e-9)

    # log features (stabilizes scale)
    for c in ["macs", "bytes_total", "cycles_est"]:
        df[f"log1p_{c}"] = np.log1p(df[c].clip(min=0))

    # ratios (useful shape structure)
    df["ratio_out_in"] = df["OUT"] / (df["IN"] + 1e-9)
    df["ratio_cout_cin"] = df["OUT"] / (df["C"] + 1e-9)

    return df

def one_hot(df: pd.DataFrame, cats=CATEGORICALS) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[c for c in cats if c in df.columns], dummy_na=True)

def make_xy(df: pd.DataFrame, target_col: str):
    df = add_derived_features(df.copy())
    df = one_hot(df)

    y = df[target_col].map(_safe_num).values.astype(np.float32)
    # predict log latency to stabilize training
    y_log = np.log1p(np.clip(y, 0.0, None))

    drop_cols = [target_col, "y_fpga_est_io_ms", "y_fpga_est_compute_ms", "y_cpu_measured_ms", "y_gpu_measured_ms", "notes"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # config_id is not a feature
    if "config_id" in X.columns:
        X = X.drop(columns=["config_id"])

    return X.values.astype(np.float32), y.astype(np.float32), y_log.astype(np.float32), list(X.columns)
