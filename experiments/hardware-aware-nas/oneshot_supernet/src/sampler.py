from __future__ import annotations
import random
from typing import Dict, Tuple, List

def sample_subnet(kernels: List[int], widths: List[int], seed: int | None = None) -> Dict[str, Tuple[int, int]]:
    if seed is not None:
        random.seed(seed)
    def pick():
        return (random.choice(kernels), random.choice(widths))
    return {"b1": pick(), "b2": pick(), "b3": pick()}
