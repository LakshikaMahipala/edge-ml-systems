from __future__ import annotations
import numpy as np

def conv1d_full_naive(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    N = x.shape[0]
    K = h.shape[0]
    y = np.zeros(N + K - 1, dtype=np.float32)
    for n in range(N + K - 1):
        s = 0.0
        kmin = max(0, n - (N - 1))
        kmax = min(K - 1, n)
        for k in range(kmin, kmax + 1):
            s += x[n - k] * h[k]
        y[n] = s
    return y
