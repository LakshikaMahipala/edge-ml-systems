from __future__ import annotations
import json
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def train_val_split(rows: List[Dict[str, Any]], val_ratio: float = 0.2, seed: int = 0):
    import random
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    nval = int(len(rows) * val_ratio)
    val_idx = set(idx[:nval])
    tr = [rows[i] for i in range(len(rows)) if i not in val_idx]
    va = [rows[i] for i in range(len(rows)) if i in val_idx]
    return tr, va
