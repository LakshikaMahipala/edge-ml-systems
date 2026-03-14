# inference_bench/src/reporting.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RunMeta:
    project: str
    script: str
    model: str
    device: str
    input_desc: str
    warmup: int
    iters: int
    platform: str
    python: str
    torch: str


@dataclass
class PerfSummary:
    preprocess: Dict[str, float]
    inference: Dict[str, float]
    postprocess: Dict[str, float]
    end_to_end: Dict[str, float]


def print_budget(summary: PerfSummary) -> None:
    """
    Print component percentiles and a simple p50 budget sanity check.
    """
    pre50 = summary.preprocess.get("p50_ms", float("nan"))
    inf50 = summary.inference.get("p50_ms", float("nan"))
    post50 = summary.postprocess.get("p50_ms", float("nan"))
    e2e50 = summary.end_to_end.get("p50_ms", float("nan"))

    print("Latency Breakdown (ms)")
    print(f"  Preprocess p50: {pre50:.4f}")
    print(f"  Inference  p50: {inf50:.4f}")
    print(f"  Postproc   p50: {post50:.4f}")
    print(f"  End-to-end p50: {e2e50:.4f}")
    print("")
    print("Sanity check (approx): preprocess_p50 + inference_p50 + postprocess_p50")
    print(f"  Sum p50 ≈ {(pre50 + inf50 + post50):.4f} ms (compare vs end-to-end p50 {e2e50:.4f} ms)")
    print("Note: Not exact due to overheads and measurement variance.")


def save_json(meta: RunMeta, perf: PerfSummary, out_dir: str = "inference_bench/results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"{meta.project}_{meta.model}_{meta.device}_{ts}.json".replace("/", "_").replace(" ", "_")
    path = os.path.join(out_dir, fname)

    payload: Dict[str, Any] = {
        "meta": asdict(meta),
        "perf": asdict(perf),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
