from __future__ import annotations
import time
import numpy as np

class BenchTimer:
    def __init__(self, warmup: int = 20, iters: int = 100):
        self.warmup = warmup
        self.iters = iters

    def run(self, fn):
        # warmup
        for _ in range(self.warmup):
            fn()
        times = []
        for _ in range(self.iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
        arr = np.array(times)
        return {
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "mean_ms": float(arr.mean()),
        }
