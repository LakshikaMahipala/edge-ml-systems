from __future__ import annotations
import numpy as np

def mae(y, yhat): return float(np.mean(np.abs(y - yhat)))
def rmse(y, yhat): return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mape(y, yhat, eps=1e-9):
    y = np.maximum(y, eps)
    return float(np.mean(np.abs((y - yhat) / y)) * 100.0)

def spearmanr(y, yhat):
    # Spearman = Pearson corr of ranks (no scipy dependency)
    def rank(a):
        temp = np.argsort(a)
        r = np.empty_like(temp, dtype=np.float32)
        r[temp] = np.arange(len(a), dtype=np.float32)
        return r
    ry = rank(y)
    rh = rank(yhat)
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    rh = (rh - rh.mean()) / (rh.std() + 1e-9)
    return float(np.mean(ry * rh))

def report(y, yhat):
    return {
        "mae_ms": mae(y, yhat),
        "rmse_ms": rmse(y, yhat),
        "mape_pct": mape(y, yhat),
        "spearman": spearmanr(y, yhat),
    }
