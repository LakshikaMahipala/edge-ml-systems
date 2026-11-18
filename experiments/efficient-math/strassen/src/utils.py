from __future__ import annotations
import time
import numpy as np

def timer_ms(fn, iters: int = 30, warmup: int = 5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1000.0)
    ts = np.array(ts, dtype=np.float64)
    return {
        "p50_ms": float(np.percentile(ts, 50)),
        "p99_ms": float(np.percentile(ts, 99)),
        "mean_ms": float(np.mean(ts)),
    }

def pad_to_pow2(A: np.ndarray) -> tuple[np.ndarray, int]:
    n = A.shape[0]
    m = 1
    while m < n:
        m *= 2
    if m == n:
        return A, n
    P = np.zeros((m, m), dtype=A.dtype)
    P[:n, :n] = A
    return P, n
