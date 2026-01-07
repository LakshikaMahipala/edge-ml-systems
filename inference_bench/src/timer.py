from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass(frozen=True)
class TimingResult:
    latencies_s: np.ndarray
    total_items: int
    total_time_s: float

    @property
    def throughput_items_per_s(self) -> float:
        return float(self.total_items) / float(self.total_time_s) if self.total_time_s > 0 else 0.0

    def percentile_ms(self, p: float) -> float:
        return float(np.percentile(self.latencies_s * 1e3, p))

    def summary(self) -> dict:
        return {
            "iters_measured": int(self.latencies_s.size),
            "total_items": int(self.total_items),
            "total_time_s": float(self.total_time_s),
            "throughput_items_per_s": float(self.throughput_items_per_s),
            "p50_ms": self.percentile_ms(50),
            "p90_ms": self.percentile_ms(90),
            "p99_ms": self.percentile_ms(99),
            "mean_ms": float(np.mean(self.latencies_s) * 1e3) if self.latencies_s.size else 0.0,
        }


def benchmark(
    step_fn: Callable[[int], int],
    *,
    warmup: int,
    iters: int,
    synchronize_fn: Optional[Callable[[], None]] = None,
) -> TimingResult:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if iters <= 0:
        raise ValueError("iters must be > 0")

    # Warmup (not recorded)
    for i in range(warmup):
        _ = int(step_fn(i))

    lat: List[float] = []
    total_items = 0

    t_all0 = time.perf_counter()
    for i in range(iters):
        t0 = time.perf_counter()
        n = int(step_fn(i))
        if synchronize_fn is not None:
            synchronize_fn()
        t1 = time.perf_counter()
        lat.append(t1 - t0)
        total_items += n
    t_all1 = time.perf_counter()

    return TimingResult(
        latencies_s=np.asarray(lat, dtype=np.float64),
        total_items=int(total_items),
        total_time_s=float(t_all1 - t_all0),
    )
