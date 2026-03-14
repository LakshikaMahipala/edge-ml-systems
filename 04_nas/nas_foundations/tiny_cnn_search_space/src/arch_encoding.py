from __future__ import annotations
from typing import Dict, Any
import json

def arch_to_json(arch: Dict[str, Any]) -> str:
    return json.dumps(arch, sort_keys=True)

def arch_id(arch: Dict[str, Any]) -> str:
    # stable identifier for logging
    return str(abs(hash(arch_to_json(arch))))

def pretty_arch(arch: Dict[str, Any]) -> str:
    return json.dumps(arch, indent=2, sort_keys=True)
