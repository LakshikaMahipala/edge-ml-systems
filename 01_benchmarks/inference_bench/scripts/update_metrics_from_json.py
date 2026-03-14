# scripts/update_metrics_from_json.py
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=str, help="Path to a saved benchmark JSON file")
    args = ap.parse_args()

    p = Path(args.json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    meta = data.get("meta", {})
    perf = data.get("perf", {})
    e2e = perf.get("end_to_end", {})

    project = meta.get("project", "inference_bench")
    model = meta.get("model", "unknown")
    device = meta.get("device", "unknown")
    input_desc = meta.get("input_desc", "unknown")
    warmup = meta.get("warmup", "unknown")
    iters = meta.get("iters", "unknown")

    p50 = e2e.get("p50_ms", "TBD")
    p99 = e2e.get("p99_ms", "TBD")

    notes = f"warmup={warmup}, iters={iters}, torch={meta.get('torch','?')}, platform={meta.get('platform','?')}"

    # Markdown row for docs/metrics.md
    row = f"| {project} | {model} | {device} | TBD | {input_desc} | {p50} | {p99} | TBD | TBD | {notes} |"
    print(row)


if __name__ == "__main__":
    main()
