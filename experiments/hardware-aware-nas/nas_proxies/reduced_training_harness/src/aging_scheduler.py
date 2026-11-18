from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class AgingConfig:
    every_k: int = 20     # re-evaluate every K new candidates
    top_m: int = 5        # re-evaluate top M
    extra_steps: int = 50 # optional: longer proxy budget during aging
