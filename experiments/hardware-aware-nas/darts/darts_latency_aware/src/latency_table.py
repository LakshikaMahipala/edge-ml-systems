from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class LatencyTable:
    """
    Proxy op latencies (arbitrary units).
    Replace later with measured microbenchmarks.
    """
    op_latency: Dict[str, float]

def default_latency_table() -> LatencyTable:
    return LatencyTable(
        op_latency={
            "skip": 0.10,
            "conv3": 1.00,
            "conv5": 1.60,
        }
    )
