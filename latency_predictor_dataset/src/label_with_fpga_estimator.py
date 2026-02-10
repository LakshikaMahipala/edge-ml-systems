from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

def bytes_per_sec_from_baud(baud: float) -> float:
    return baud / 10.0

def io_time_ms(total_bytes: int, baud: float) -> float:
    return (total_bytes / bytes_per_sec_from_baud(baud)) * 1000.0

def compute_time_ms(cycles: int, f_clk_mhz: float) -> float:
    return (cycles / (f_clk_mhz * 1e6)) * 1000.0

def fc_bytes(IN: int, OUT: int) -> Tuple[int, int]:
    x = IN
    w = OUT * IN
    b = OUT * 4
    return (x + w + b), OUT

def dwconv1d_bytes(C: int, L: int, K: int) -> Tuple[int, int]:
    x = C * L
    w = C * K
    b = C * 4
    return (x + w + b), (C * (L - K + 1))

def fc_cycles(IN: int, c0: int = 0) -> int:
    return IN + c0

def dwconv_cycles(C: int, L: int, K: int, c1: int = 0) -> int:
    return C * (L - K + 1) * K + c1

def label_row(row: Dict[str, Any], baud: float, f_clk_mhz: float) -> Dict[str, Any]:
    op = row["op_type"]
    if op == "INT8_FC":
        b_in, b_out = fc_bytes(int(row["IN"]), int(row["OUT"]))
        cyc = fc_cycles(int(row["IN"]), 0)
        macs = int(row["IN"]) * int(row["OUT"])
    elif op == "INT8_DWCONV1D_K3":
        C = int(row["C"]); L = int(row["L"]); K = int(row.get("K", 3))
        b_in, b_out = dwconv1d_bytes(C, L, K)
        cyc = dwconv_cycles(C, L, K, 0)
        macs = C * (L - K + 1) * K
    else:
        raise ValueError(f"Unsupported op_type: {op}")

    bytes_total = b_in + b_out
    t_io = io_time_ms(bytes_total, baud)
    t_compute = compute_time_ms(cyc, f_clk_mhz)
    t_total = t_io + t_compute

    out = dict(row)
    out.update({
        "bytes_in": b_in,
        "bytes_out": b_out,
        "bytes_total": bytes_total,
        "macs": macs,
        "cycles_est": cyc,
        "interface": "uart",
        "baud": baud,
        "f_clk_mhz": f_clk_mhz,
        "y_fpga_est_io_ms": t_io,
        "y_fpga_est_compute_ms": t_compute,
        "y_fpga_est_total_ms": t_total,
        "y_cpu_measured_ms": "",
        "y_gpu_measured_ms": "",
        "notes": "",
    })
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", type=str, default="data/configs.jsonl")
    ap.add_argument("--out_jsonl", type=str, default="data/labeled.jsonl")
    ap.add_argument("--baud", type=float, default=115200.0)
    ap.add_argument("--f_clk_mhz", type=float, default=100.0)
    args = ap.parse_args()

    in_p = Path(args.in_jsonl)
    out_p = Path(args.out_jsonl)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with in_p.open("r", encoding="utf-8") as fi, out_p.open("w", encoding="utf-8") as fo:
        for line in fi:
            row = json.loads(line)
            labeled = label_row(row, args.baud, args.f_clk_mhz)
            fo.write(json.dumps(labeled) + "\n")

    print("Wrote labeled dataset:", out_p)

if __name__ == "__main__":
    main()
