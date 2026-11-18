from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

def set_seed(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    torch.manual_seed(seed)

def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))

def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
