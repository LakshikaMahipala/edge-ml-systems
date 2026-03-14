from __future__ import annotations
from typing import Dict, Any
import math

def bytes_per_sec_from_baud(baud: float) -> float:
    return baud / 10.0

def io_time_ms(total_bytes: int, baud: float) -> float:
    return (total_bytes / bytes_per_sec_from_baud(baud)) * 1000.0

def compute_time_ms(cycles: float, f_clk_mhz: float) -> float:
    return (cycles / (f_clk_mhz * 1e6)) * 1000.0

def _safe_int(x, d=0):
    try:
        return int(x)
    except Exception:
        return d

def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d

def estimate_point(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    v1 estimator supports multiple 'variant' types:
    - BASE_INT8_FC
    - LOWRANK_INT8_FC
    - BASE_INT8_DWCONV1D_K3
    (Winograd/FFT are modeled as transform overhead placeholders, not full kernels yet.)
    """
    variant = row.get("variant", row.get("op_type", "BASE_INT8_FC"))
    baud = _safe_float(row.get("baud", 115200))
    f_clk = _safe_float(row.get("f_clk_mhz", 100))
    II = _safe_int(row.get("II", 1))

    # Defaults
    macs = 0
    cycles_compute = 0.0
    cycles_transform = 0.0
    bytes_in = 0
    bytes_out = 0
    dsp = 0
    bram = 0

    if variant in ["BASE_INT8_FC", "INT8_FC"]:
        IN = _safe_int(row.get("IN"))
        OUT = _safe_int(row.get("OUT"))
        UNROLL = max(_safe_int(row.get("UNROLL", 1)), 1)

        # bytes: x + W + bias + output
        bytes_in = IN + (IN * OUT) + (OUT * 4)
        bytes_out = OUT

        macs = IN * OUT
        cycles_compute = (OUT * (IN / UNROLL)) * II
        dsp = UNROLL
        bram = int(math.ceil((IN * OUT) / 4096))  # crude proxy: 4KB tiles

    elif variant == "LOWRANK_INT8_FC":
        IN = _safe_int(row.get("IN"))
        OUT = _safe_int(row.get("OUT"))
        r = max(_safe_int(row.get("rank_r", 64)), 1)
        UNROLL = max(_safe_int(row.get("UNROLL", 1)), 1)

        # Two matrices: A (IN x r), B (r x OUT)
        bytes_in = IN + (IN * r) + (r * OUT) + (OUT * 4)
        bytes_out = OUT

        macs = IN * r + r * OUT

        # two stages compute; assume same UNROLL applies as MAC lanes
        cycles_compute = ((r * (IN / UNROLL)) + (OUT * (r / UNROLL))) * II

        # intermediate vector of size r (read/write) adds bandwidth pressure
        # Model as transform-like overhead in cycles (very rough)
        cycles_transform = r * 2  # write + read, in "cycle units" proxy

        dsp = UNROLL
        bram = int(math.ceil((IN * r + r * OUT) / 4096)) + int(math.ceil(r / 256))

    elif variant in ["BASE_INT8_DWCONV1D_K3", "INT8_DWCONV1D_K3"]:
        C = _safe_int(row.get("C"))
        L = _safe_int(row.get("L"))
        K = _safe_int(row.get("K", 3))
        UNROLL_C = max(_safe_int(row.get("UNROLL_C", 1)), 1)

        bytes_in = (C * L) + (C * K) + (C * 4)
        bytes_out = C * max(L - K + 1, 0)

        macs = C * max(L - K + 1, 0) * K
        cycles_compute = ((C / UNROLL_C) * max(L - K + 1, 0) * K) * II

        dsp = UNROLL_C
        bram = int(math.ceil((C * L) / 2048))

    else:
        raise ValueError(f"Unknown variant: {variant}")

    bytes_total = bytes_in + bytes_out

    t_io = io_time_ms(bytes_total, baud)
    t_compute = compute_time_ms(cycles_compute, f_clk)
    t_transform = compute_time_ms(cycles_transform, f_clk)
    t_total = t_io + t_compute + t_transform

    out = dict(row)
    out.update({
        "variant": variant,
        "bytes_in": int(bytes_in),
        "bytes_out": int(bytes_out),
        "bytes_total": int(bytes_total),
        "macs": int(macs),
        "cycles_est": float(cycles_compute),
        "transform_cycles_est": float(cycles_transform),
        "resource_proxy_dsp": int(dsp),
        "resource_proxy_bram": int(bram),
        "y_fpga_est_io_ms": float(t_io),
        "y_fpga_est_compute_ms": float(t_compute),
        "y_fpga_est_transform_ms": float(t_transform),
        "y_fpga_est_total_ms": float(t_total),
    })
    return out
