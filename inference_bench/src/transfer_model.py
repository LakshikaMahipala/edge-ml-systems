# inference_bench/src/transfer_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TransferLink:
    """
    A simplified model of a data-transfer link (PCIe, shared memory, etc.).
    bandwidth_gbps: effective bandwidth in GB/s (not Gbit/s)
    overhead_us: fixed overhead per transfer (driver/sync/call overhead)
    """
    name: str
    bandwidth_gbps: float
    overhead_us: float = 20.0


def bytes_for_tensor(n: int, c: int, h: int, w: int, dtype_bytes: int = 4) -> int:
    return int(n) * int(c) * int(h) * int(w) * int(dtype_bytes)


def estimate_transfer_ms(num_bytes: int, link: TransferLink) -> float:
    """
    Estimate transfer time in ms as:
      overhead + bytes / bandwidth
    where bandwidth is GB/s.
    """
    if link.bandwidth_gbps <= 0:
        raise ValueError("bandwidth_gbps must be > 0")

    overhead_ms = link.overhead_us / 1000.0
    data_ms = (num_bytes / (1024.0**3)) / link.bandwidth_gbps * 1000.0
    return overhead_ms + data_ms


# Reasonable default "effective" bandwidths (order-of-magnitude, not peak spec).
PCIE3_X16 = TransferLink(name="PCIe3 x16 (effective)", bandwidth_gbps=12.0, overhead_us=20.0)
PCIE4_X16 = TransferLink(name="PCIe4 x16 (effective)", bandwidth_gbps=24.0, overhead_us=20.0)
SHARED_MEM = TransferLink(name="Shared memory copy", bandwidth_gbps=20.0, overhead_us=5.0)
