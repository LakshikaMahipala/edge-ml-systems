from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Candidate:
    model_id: str
    acc: float          # higher is better
    latency_ms: float   # lower is better
    energy_mj: float    # lower is better (proxy ok)
