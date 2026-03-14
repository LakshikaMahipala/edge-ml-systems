from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class ProxyScore:
    arch_id: str
    score: float
    val_acc: float
    train_loss: float
    steps: int
    seed: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
