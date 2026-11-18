from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class ChoiceSpace:
    kernels: List[int] = (3, 5)
    widths: List[int] = (16, 24)  # output channels choices
