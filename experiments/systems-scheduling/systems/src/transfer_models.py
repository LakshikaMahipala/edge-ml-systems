from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Link:
    name: str
    bandwidth_GBps: float   # effective bandwidth
    overhead_us: float      # fixed per-transfer overhead (us)

def transfer_time_us(payload_bytes: int, link: Link) -> float:
    bw_Bps = link.bandwidth_GBps * (1024**3)
    return link.overhead_us + (payload_bytes / bw_Bps) * 1e6
