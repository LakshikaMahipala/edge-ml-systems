# inference_bench/src/streaming_bench.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from inference_bench.src.pipeline import ItemTiming


@dataclass
class StreamingSummary:
    count: int
    e2e_p50_ms: float
    e2e_p99_ms: float
    queue_p50_ms: float
    queue_p99_ms: float
    service_p50_ms: float
    service_p99_ms: float
    throughput_items_s: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "e2e_p50_ms": self.e2e_p50_ms,
            "e2e_p99_ms": self.e2e_p99_ms,
            "queue_p50_ms": self.queue_p50_ms,
            "queue_p99_ms": self.queue_p99_ms,
            "service_p50_ms": self.service_p50_ms,
            "service_p99_ms": self.service_p99_ms,
            "throughput_items_s": self.throughput_items_s,
        }


def _pct_ms(vals_s: List[float], p: float) -> float:
    if not vals_s:
        return 0.0
    arr = np.array(vals_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, p))


def summarize_streaming(timings: List[ItemTiming], wall_time_s: float) -> StreamingSummary:
    e2e = [t.e2e_s for t in timings]
    qwait = [t.queue_wait_s for t in timings]
    svc = [t.service_s for t in timings]
    n = len(timings)

    thr = (n / wall_time_s) if wall_time_s > 0 and n > 0 else 0.0

    return StreamingSummary(
        count=n,
        e2e_p50_ms=_pct_ms(e2e, 50),
        e2e_p99_ms=_pct_ms(e2e, 99),
        queue_p50_ms=_pct_ms(qwait, 50),
        queue_p99_ms=_pct_ms(qwait, 99),
        service_p50_ms=_pct_ms(svc, 50),
        service_p99_ms=_pct_ms(svc, 99),
        throughput_items_s=float(thr),
    )
