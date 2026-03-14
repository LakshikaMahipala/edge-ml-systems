from __future__ import annotations
from typing import List, Dict, Any
import json
from pathlib import Path
from heft import ScheduledTask, makespan

def export_json(path: str, sched: List[ScheduledTask]) -> None:
    out = {
        "makespan_ms": makespan(sched),
        "tasks": [
            {"task_id": t.task_id, "device": t.device, "start_ms": t.start, "end_ms": t.end}
            for t in sched
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(out, indent=2))
