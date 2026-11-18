"""Bit-exact Python reference ("golden model") for the INT8 RTL kernels.

Every function here mirrors one in rtl/fxp_pkg.sv or a kernel in rtl/.
The testbenches compare the hardware against vectors produced from this file,
so any mismatch points at the RTL, not at the reference.

Integer conventions (same as the RTL):
  * inputs and weights: int8, two's complement
  * accumulator and bias: int32
  * requantization: y = clip( (acc + 2**(shift-1)) >> shift , -128, 127 )
    ``>>`` on negative numbers is an arithmetic (floor) shift in both Python and
    SystemVerilog's ``>>>``, so the rounding is "round half up" for all signs.
"""
from __future__ import annotations

import numpy as np

INT8_MIN, INT8_MAX = -128, 127
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1


def rshift_round(x, shift: int):
    """Arithmetic right shift with round-half-up. Works on ints and int64 arrays."""
    x = np.asarray(x, dtype=np.int64)
    if shift <= 0:
        return x
    return (x + (1 << (shift - 1))) >> shift


def saturate(x, bits: int):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def requant_int8(acc, shift: int) -> np.ndarray:
    return saturate(rshift_round(acc, shift), 8).astype(np.int8)


def _check_int32(acc: np.ndarray) -> None:
    if acc.min() < INT32_MIN or acc.max() > INT32_MAX:
        raise OverflowError("accumulator exceeds int32: the RTL would wrap around")


def fxp_mul(a, b, frac_bits: int, bits: int = 8) -> np.ndarray:
    prod = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    return saturate(rshift_round(prod, frac_bits), bits)


def fxp_add(a, b, bits: int = 8) -> np.ndarray:
    return saturate(np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64), bits)


def fc_int8(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """Fully-connected layer. x: (N, IN) int8, w: (OUT, IN) int8, b: (OUT,) int32 -> (N, OUT) int8."""
    acc = x.astype(np.int64) @ w.astype(np.int64).T + b.astype(np.int64)
    _check_int32(acc)
    return requant_int8(acc, shift)


def dwconv1d_int8(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """Depthwise 1D conv, valid padding, stride 1.
    x: (N, C, L) int8, w: (C, K) int8, b: (C,) int32 -> (N, C, L-K+1) int8."""
    n, c, length = x.shape
    k = w.shape[1]
    lout = length - k + 1
    acc = np.zeros((n, c, lout), dtype=np.int64) + b.astype(np.int64)[None, :, None]
    for tap in range(k):
        acc += x[:, :, tap:tap + lout].astype(np.int64) * w[None, :, tap, None].astype(np.int64)
    _check_int32(acc)
    return requant_int8(acc, shift)


# ---------------------------------------------------------------------------
# From float to INT8: how real weights become the integers the hardware uses
# ---------------------------------------------------------------------------

def quantize_symmetric(t: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor quantization: t ≈ q * scale, q in [-(2^(b-1)-1), 2^(b-1)-1]."""
    qmax = (1 << (bits - 1)) - 1
    scale = float(np.max(np.abs(t))) / qmax or 1.0
    q = np.clip(np.round(t / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def quantize_fc_layer(w_f: np.ndarray, b_f: np.ndarray, x_scale: float, y_absmax: float):
    """Quantize a float FC layer for int8_fc with a power-of-two output scale.

    Integer accumulator units are acc_scale = x_scale * w_scale, so the bias is
    stored in those units. After ``>> shift`` the output scale is
    y_scale = acc_scale * 2**shift. ``y_absmax`` (the largest float output seen
    on calibration data) picks the smallest shift that avoids saturation, which
    keeps the most output resolution.
    Returns (w_q, b_q, shift, y_scale).
    """
    w_q, w_scale = quantize_symmetric(w_f)
    acc_scale = x_scale * w_scale
    shift = max(0, int(np.ceil(np.log2(y_absmax / (INT8_MAX * acc_scale)))))
    b_q = np.round(b_f / acc_scale).astype(np.int64)
    if np.abs(b_q).max(initial=0) > INT32_MAX:
        raise OverflowError("bias does not fit in int32")
    return w_q, b_q.astype(np.int32), shift, acc_scale * (1 << shift)
