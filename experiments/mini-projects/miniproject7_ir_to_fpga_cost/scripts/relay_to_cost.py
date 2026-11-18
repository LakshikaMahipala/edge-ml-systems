from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", type=str, default="", help="Direct FPGA op graph JSON")
    ap.add_argument("--out_json", type=str, default="results/cost.json")
    ap.add_argument("--baud", type=float, default=115200.0)
    ap.add_argument("--f_clk_mhz", type=float, default=100.0)
    ap.add_argument("--host_ms", type=float, default=0.0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    estimator = root.parent / "fpga_cost_model" / "src" / "fpga_latency_estimator_v0.py"

    if args.graph_json:
        graph_path = Path(args.graph_json)
    else:
        graph_path = root / "examples" / "graph_from_relay_example.json"

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(estimator),
        "--graph_json", str(graph_path),
        "--baud", str(args.baud),
        "--f_clk_mhz", str(args.f_clk_mhz),
        "--host_ms", str(args.host_ms),
        "--out", str(out_path),
    ]

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Saved cost JSON:", out_path)

if __name__ == "__main__":
    main()
