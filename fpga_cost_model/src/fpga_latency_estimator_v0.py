from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from ops import fc_bytes, dwconv1d_bytes

def bytes_per_sec_from_baud(baud: float) -> float:
    return baud / 10.0

def io_time_ms(total_bytes: int, baud: float) -> float:
    bps = bytes_per_sec_from_baud(baud)
    return (total_bytes / bps) * 1000.0

def compute_time_ms(cycles: int, f_clk_hz: float) -> float:
    return (cycles / f_clk_hz) * 1000.0

def estimate_cycles(op: str, params: Dict[str, Any], overhead: Dict[str, int]) -> int:
    if op == "INT8_FC":
        IN = int(params["IN"])
        c0 = int(overhead.get("fc_c0", 0))
        return IN + c0
    if op == "INT8_DWCONV1D_K3":
        C = int(params["C"]); L = int(params["L"]); K = int(params.get("K", 3))
        c1 = int(overhead.get("dw_c1", 0))
        return C * (L - K + 1) * K + c1
    if op == "RELU_INT8":
        # assume negligible compute for v0
        return int(overhead.get("relu_c", 0))
    raise ValueError(f"Unknown op: {op}")

def estimate_bytes(op: str, params: Dict[str, Any]) -> tuple[int, int]:
    if op == "INT8_FC":
        IN = int(params["IN"]); OUT = int(params["OUT"])
        b_in, b_out, _ = fc_bytes(IN, OUT)
        return b_in, b_out
    if op == "INT8_DWCONV1D_K3":
        C = int(params["C"]); L = int(params["L"]); K = int(params.get("K", 3))
        b_in, b_out, _ = dwconv1d_bytes(C, L, K)
        return b_in, b_out
    if op == "RELU_INT8":
        # in-place by default; no extra I/O
        N = int(params["N"])
        return N, N
    raise ValueError(f"Unknown op: {op}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_json", type=str, required=True, help="List of ops with params")
    ap.add_argument("--baud", type=float, default=115200.0)
    ap.add_argument("--f_clk_mhz", type=float, default=100.0)
    ap.add_argument("--host_ms", type=float, default=0.0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--fc_c0", type=int, default=0, help="FC fixed cycle overhead")
    ap.add_argument("--dw_c1", type=int, default=0, help="DWConv fixed cycle overhead")
    ap.add_argument("--relu_c", type=int, default=0, help="ReLU cycles (usually 0)")
    args = ap.parse_args()

    graph = json.loads(Path(args.graph_json).read_text(encoding="utf-8"))
    f_clk_hz = args.f_clk_mhz * 1e6
    overhead = {"fc_c0": args.fc_c0, "dw_c1": args.dw_c1, "relu_c": args.relu_c}

    op_rows: List[Dict[str, Any]] = []
    total_bytes = 0
    total_cycles = 0

    for node in graph:
        op = node["op"]
        params = node.get("params", {})
        b_in, b_out = estimate_bytes(op, params)
        cyc = estimate_cycles(op, params, overhead)

        total_bytes += (b_in + b_out)   # UART mode A: send inputs/weights + receive outputs
        total_cycles += cyc

        op_rows.append({
            "op": op,
            "params": params,
            "bytes_in": b_in,
            "bytes_out": b_out,
            "cycles": cyc,
        })

    t_io = io_time_ms(total_bytes, args.baud)
    t_compute = compute_time_ms(total_cycles, f_clk_hz)
    t_total = t_io + t_compute + args.host_ms

    summary = {
        "assumptions": {
            "baud": args.baud,
            "bytes_per_sec": bytes_per_sec_from_baud(args.baud),
            "f_clk_mhz": args.f_clk_mhz,
            "host_ms": args.host_ms,
            "overhead_cycles": overhead,
        },
        "totals": {
            "total_bytes": total_bytes,
            "total_cycles": total_cycles,
            "t_io_ms": t_io,
            "t_compute_ms": t_compute,
            "t_total_ms": t_total,
        },
        "ops": op_rows,
    }

    print("FPGA Latency Estimator v0")
    print(f"baud={args.baud} (~{bytes_per_sec_from_baud(args.baud):.1f} B/s), f_clk={args.f_clk_mhz} MHz")
    print(f"total_bytes={total_bytes}, total_cycles={total_cycles}")
    print(f"T_io_ms={t_io:.3f}, T_compute_ms={t_compute:.6f}, host_ms={args.host_ms:.3f}")
    print(f"T_total_ms={t_total:.3f}")

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Saved:", out_p)

if __name__ == "__main__":
    main()
