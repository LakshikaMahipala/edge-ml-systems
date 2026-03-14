from __future__ import annotations
import json
from typing import Dict, Any, List, Tuple

def load_registry(path: str) -> List[Dict[str, Any]]:
    return json.loads(open(path, "r", encoding="utf-8").read())["entries"]

def build_cost_table(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    cost_table[kernel_id][device] = latency_ms
    """
    out: Dict[str, Dict[str, float]] = {}
    for e in entries:
        k = e["kernel_id"]
        d = e["device"]
        out.setdefault(k, {})[d] = float(e["latency_p50_ms"])
    return out

def ensure_all_devices(cost_table: Dict[str, Dict[str, float]], devices: List[str], default_big: float = 1e6):
    for k in cost_table:
        for d in devices:
            cost_table[k].setdefault(d, default_big)
