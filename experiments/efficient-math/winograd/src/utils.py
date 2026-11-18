from __future__ import annotations
import time
import numpy as np

def timer_ms(fn, iters: int = 50, warmup: int = 10):
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
