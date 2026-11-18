from __future__ import annotations
from typing import Dict, Any
import math

def latency_table_units(op: str) -> float:
    return {"skip": 0.10, "conv3": 1.00, "conv5": 1.60}.get(op, 1.0)

def fpga_cycles_proxy(case: Dict[str, Any], P: int = 64) -> float:
    """
    Very rough: ops / P then log.
    For conv: MACs = cout*h*w*(cin)*k*k
    For skip: tiny.
    """
    op = case["op"]
    if op == "skip":
        ops = 1.0
    else:
        cin, cout, h, w, k = case["cin"], case["cout"], case["h"], case["w"], case["k"]
        ops = float(cout * h * w * cin * k * k)
    cycles = ops / float(P)
    return float(math.log(cycles + 1.0))

def predict(case: Dict[str, Any], w_table: float = 1.0, w_cycles: float = 1.0) -> float:
    return float(w_table * latency_table_units(case["op"]) + w_cycles * fpga_cycles_proxy(case))
