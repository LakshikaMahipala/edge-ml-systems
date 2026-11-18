from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np
import time


@dataclass
class LatencyStats:
    max_hist: int = 2000
    lat_ms: List[float] = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def add(self, v_ms: float) -> None:
        self.lat_ms.append(float(v_ms))
        if len(self.lat_ms) > self.max_hist:
            self.lat_ms.pop(0)

    def summary(self) -> dict:
        if not self.lat_ms:
            return {"count": 0}
        a = np.array(self.lat_ms, dtype=np.float64)
        return {
            "count": int(len(a)),
            "mean_ms": float(a.mean()),
            "p50_ms": float(np.percentile(a, 50)),
            "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max()),
        }
