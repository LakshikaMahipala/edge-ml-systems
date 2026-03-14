from __future__ import annotations
from pathlib import Path
import json
from kernels_to_measure import default_cases, to_records

def main():
    Path("data").mkdir(exist_ok=True)
    tmpl = []
    for rec in to_records(default_cases()):
        rec.update({
            "fpga": {"board": "TBD", "clock_mhz": 0},
            "latency_p50_us": 0.0,
            "latency_p99_us": 0.0,
            "runs": 0,
            "warmup": 0,
        })
        tmpl.append(rec)

    p = Path("data/template_measurements.jsonl")
    with p.open("w", encoding="utf-8") as f:
        for r in tmpl:
            f.write(json.dumps(r) + "\n")
    print("wrote:", p)

if __name__ == "__main__":
    main()
