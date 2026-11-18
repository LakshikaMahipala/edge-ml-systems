from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict
import random
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

def append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

def write_json(path: str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))
