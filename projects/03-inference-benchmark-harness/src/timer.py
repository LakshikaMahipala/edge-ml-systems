# inference_bench/src/timer.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class TimingResult:
    warmup_iters: int
    measure_iters: int
    latencies_ms: List[float]

    def percentile(self, p: float) -> float:
        """
        Compute percentile p (0..100) using linear interpolation.
        No numpy dependency (keeps repo lightweight).
        """
        if not self.latencies_ms:
            raise ValueError("No latencies recorded.")
        if p < 0 or p > 100:
            raise ValueError("Percentile p must be between 0 and 100.")

        xs = sorted(self.latencies_ms)
        n = len(xs)
        if n == 1:
            return xs[0]

        # Position in [0, n-1]
        pos = (p / 100.0) * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    def summary(self) -> Dict[str, float]:
        return {
            "p50_ms": self.percentile(50),
            "p90_ms": self.percentile(90),
            "p99_ms": self.percentile(99),
            "min_ms": min(self.latencies_ms),
            "max_ms": max(self.latencies_ms),
        }


class Timer:
    """
    Minimal, correct timer for CPU-side benchmarking.
    - Uses perf_counter() for high-resolution monotonic timing.
    - Warmup is excluded from measurement.
    """

    def __init__(
        self,
        warmup_iters: int = 30,
        measure_iters: int = 200,
        sync: Optional[Callable[[], None]] = None,
    ) -> None:
        """
        sync: optional function called before and after timing fn()
              (useful later for GPU sync; today keep None)
        """
        if warmup_iters < 0 or measure_iters <= 0:
            raise ValueError("warmup_iters must be >=0 and measure_iters must be >0")
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.sync = sync

    def run(self, fn: Callable[[], None]) -> TimingResult:
        # Warmup (not measured)
        for _ in range(self.warmup_iters):
            fn()

        latencies_ms: List[float] = []
        for _ in range(self.measure_iters):
            if self.sync:
                self.sync()
            t0 = time.perf_counter()
            fn()
            if self.sync:
                self.sync()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        return TimingResult(
            warmup_iters=self.warmup_iters,
            measure_iters=self.measure_iters,
            latencies_ms=latencies_ms,
        )
