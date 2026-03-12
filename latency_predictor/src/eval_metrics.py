from __future__ import annotations
from typing import List
import math

def spearman(x: List[float], y: List[float]) -> float:
    assert len(x) == len(y)
    n = len(x)
    if n < 2:
        return 0.0

    # ranks (1=best). For latency lower is better.
    def rank(vals):
        idx = sorted(range(n), key=lambda i: vals[i])
        r = [0]*n
        for k,i in enumerate(idx):
            r[i] = k+1
        return r

    rx = rank(x)
    ry = rank(y)
    d2 = 0.0
    for i in range(n):
        d = rx[i] - ry[i]
        d2 += d*d
    return 1.0 - (6.0*d2)/(n*(n*n-1.0))

def mape(y_true: List[float], y_pred: List[float]) -> float:
    s = 0.0
    for t,p in zip(y_true, y_pred):
        s += abs(p-t)/max(t, 1e-9)
    return s/len(y_true)

def rmse(y_true: List[float], y_pred: List[float]) -> float:
    s2 = 0.0
    for t,p in zip(y_true, y_pred):
        s2 += (p-t)*(p-t)
    return math.sqrt(s2/len(y_true))
