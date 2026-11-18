from __future__ import annotations
from typing import List
from metrics_schema import Candidate

def dominates(a: Candidate, b: Candidate) -> bool:
    """
    a dominates b if:
    - acc_a >= acc_b
    - lat_a <= lat_b
    - ene_a <= ene_b
    and at least one strict improvement.
    """
    cond_all = (
        a.acc >= b.acc and
        a.latency_ms <= b.latency_ms and
        a.energy_mj <= b.energy_mj
    )
    cond_strict = (
        a.acc > b.acc or
        a.latency_ms < b.latency_ms or
        a.energy_mj < b.energy_mj
    )
    return cond_all and cond_strict

def pareto_front(cands: List[Candidate]) -> List[Candidate]:
    front = []
    for c in cands:
        dominated = False
        for other in cands:
            if other.model_id != c.model_id and dominates(other, c):
                dominated = True
                break
        if not dominated:
            front.append(c)
    # sort for readability: lowest latency then highest acc
    front.sort(key=lambda x: (x.latency_ms, -x.acc))
    return front
