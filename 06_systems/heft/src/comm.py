from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class LinkModel:
    bandwidth_GBps: float
    overhead_ms: float

def comm_ms(bytes_: int, same_device: bool, link: LinkModel) -> float:
    if same_device:
        return 0.0
    bw_Bps = link.bandwidth_GBps * (1024**3)
    return link.overhead_ms + (bytes_ / bw_Bps) * 1e3
