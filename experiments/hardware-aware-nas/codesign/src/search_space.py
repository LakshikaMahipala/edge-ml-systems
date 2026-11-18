from __future__ import annotations
from typing import List, Dict, Any

def arch_space() -> List[Dict[str, Any]]:
    out = []
    for block in ["conv3", "conv5"]:
        for width in [16, 24]:
            for depth in [2, 3]:
                out.append({"block": block, "width": width, "depth": depth})
    return out

def hw_space() -> List[Dict[str, Any]]:
    out = []
    for P in [8, 16, 32, 64]:
        for tile in [8, 16, 32]:
            for ii_factor in [1.0, 1.2]:
                out.append({"P": P, "tile": tile, "ii_factor": ii_factor})
    return out

def joint_space() -> List[Dict[str, Any]]:
    out = []
    for a in arch_space():
        for h in hw_space():
            out.append({"arch": a, "hw": h})
    return out
