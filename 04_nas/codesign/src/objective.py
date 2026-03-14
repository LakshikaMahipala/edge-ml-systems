from __future__ import annotations
from typing import Dict, Any
import math

def acc_proxy(arch: Dict[str, Any]) -> float:
    # capacity proxy (higher is better)
    block = arch["block"]
    width = arch["width"]
    depth = arch["depth"]
    base = width * depth
    if block == "conv5":
        base *= 1.15
    return float(1.0 - math.exp(-0.02 * base))

def latency_proxy(arch: Dict[str, Any], hw: Dict[str, Any]) -> float:
    # ops proxy scaled by parallelism and ii_factor
    block = arch["block"]
    width = arch["width"]
    depth = arch["depth"]
    k = 5 if block == "conv5" else 3
    ops = float(depth * width * width * k * k)  # crude
    P = float(hw["P"])
    ii = float(hw["ii_factor"])
    return float(math.log((ops / P) * ii + 1.0))

def energy_proxy(arch: Dict[str, Any], hw: Dict[str, Any]) -> float:
    # energy proxy ~ latency * power_proxy, power rises with P
    lat = latency_proxy(arch, hw)
    P = float(hw["P"])
    power = math.log(P + 1.0)
    return float(lat * power)
