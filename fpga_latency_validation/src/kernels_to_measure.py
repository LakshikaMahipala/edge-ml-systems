from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass(frozen=True)
class KernelCase:
    case_id: str
    op: str           # "skip" | "conv3" | "conv5"
    cin: int
    cout: int
    h: int
    w: int
    k: int
    dtype: str        # "int8" etc.
    note: str

def default_cases() -> List[KernelCase]:
    return [
        KernelCase("skip_32", "skip", 16, 16, 32, 32, 1, "int8", "bypass/copy path proxy"),
        KernelCase("conv3_16", "conv3", 16, 16, 32, 32, 3, "int8", "代表 conv3"),
        KernelCase("conv5_16", "conv5", 16, 16, 32, 32, 5, "int8", "代表 conv5"),
        KernelCase("conv3_24", "conv3", 24, 24, 32, 32, 3, "int8", "width variant"),
        KernelCase("conv5_24", "conv5", 24, 24, 32, 32, 5, "int8", "width variant"),
    ]

def to_records(cases: List[KernelCase]) -> List[Dict[str, Any]]:
    return [asdict(c) for c in cases]
