from __future__ import annotations
import json
from typing import Iterator, Dict, Any

def load_candidates(jsonl_path: str) -> Iterator[Dict[str, Any]]:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # expected format: {"arch_id":..., "arch":...}
            yield rec
