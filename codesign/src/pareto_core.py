from __future__ import annotations
from typing import Dict, Any, List

def dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """
    Objectives:
    - acc: higher better
    - latency: lower better
    - energy: lower better
    """
    cond_all = (
        a["acc"] >= b["acc"] and
        a["latency"] <= b["latency"] and
        a["energy"] <= b["energy"]
    )
    cond_strict = (
        a["acc"] > b["acc"] or
        a["latency"] < b["latency"] or
        a["energy"] < b["energy"]
    )
    return cond_all and cond_strict

def pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front = []
    for p in points:
        dom = False
        for q in points:
            if q is p:
                continue
            if dominates(q, p):
                dom = True
                break
        if not dom:
            front.append(p)
    # readable order: lowest latency then highest acc
    front.sort(key=lambda x: (x["latency"], -x["acc"]))
    return front
