# inference_bench/src/memory.py
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Unix-only reliable approach: resource.getrusage
try:
    import resource
except Exception:
    resource = None  # type: ignore


@dataclass
class MemoryResult:
    peak_rss_mb: float
    note: str


def get_peak_rss_mb() -> MemoryResult:
    """
    Returns peak resident set size (RSS) in MB.
    - On Linux, ru_maxrss is in KB.
    - On macOS, ru_maxrss is in bytes.
    """
    if resource is None:
        return MemoryResult(peak_rss_mb=-1.0, note="resource module not available")

    usage = resource.getrusage(resource.RUSAGE_SELF)
    ru = float(usage.ru_maxrss)

    # Detect platform units
    if sys.platform == "darwin":
        # bytes -> MB
        peak_mb = ru / (1024.0 * 1024.0)
        note = "macOS: ru_maxrss bytes"
    else:
        # assume Linux: KB -> MB
        peak_mb = ru / 1024.0
        note = "Linux: ru_maxrss KB"

    return MemoryResult(peak_rss_mb=peak_mb, note=note)
