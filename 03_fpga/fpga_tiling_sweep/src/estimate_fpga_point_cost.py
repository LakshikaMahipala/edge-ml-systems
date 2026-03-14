from __future__ import annotations
from typing import Dict, Any, Tuple

def bytes_per_sec_from_baud(baud: float) -> float:
    return baud / 10.0

def io_time_ms(total_bytes: int, baud: float) -> float:
    return (total_bytes / bytes_per_sec_from_baud(baud)) * 1000.0

def compute_time_ms(cycles: float, f_clk_mhz: float) -> float:
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

def estimate(row: Dict[str, Any]) -> Dict[str, Any]:
    op = row["op_type"]
    baud = float(row["baud"])
    f_clk = float(row["f_clk_mhz"])
    II = int(row.get("II", 1))

    if op == "INT8_FC":
        IN = int(row["IN"]); OUT = int(row["OUT"])
        UNROLL = int(row["UNROLL"])
        b_in, b_out = fc_bytes(IN, OUT)
        macs = IN * OUT

        # cycles: OUT outputs, each needs IN MACs, UNROLL MACs per cycle, multiplied by II
        cycles = (OUT * (IN / max(UNROLL, 1.0))) * II

        resource_proxy_dsp = UNROLL  # crude: 1 dsp per lane
    elif op == "INT8_DWCONV1D_K3":
        C = int(row["C"]); L = int(row["L"]); K = int(row.get("K", 3))
        UNROLL_C = int(row["UNROLL_C"])
        b_in, b_out = dwconv1d_bytes(C, L, K)
        macs = C * (L - K + 1) * K

        # cycles: per output element, K macs; parallel over channels by UNROLL_C; multiplied by II
        cycles = ((C / max(UNROLL_C, 1.0)) * (L - K + 1) * K) * II

        resource_proxy_dsp = UNROLL_C
    else:
        raise ValueError(f"Unknown op_type: {op}")

    bytes_total = b_in + b_out
    t_io = io_time_ms(bytes_total, baud)
    t_compute = compute_time_ms(cycles, f_clk)
    t_total = t_io + t_compute

    out = dict(row)
    out.update({
        "bytes_in": int(b_in),
        "bytes_out": int(b_out),
        "bytes_total": int(bytes_total),
        "macs": int(macs),
        "cycles_est": float(cycles),
        "resource_proxy_dsp": int(resource_proxy_dsp),
        "y_fpga_est_io_ms": float(t_io),
        "y_fpga_est_compute_ms": float(t_compute),
        "y_fpga_est_total_ms": float(t_total),
    })
    return out
