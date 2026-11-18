from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Literal

OpType = Literal["INT8_FC", "INT8_DWCONV1D_K3", "RELU_INT8"]

@dataclass
class OpSpec:
    op: OpType
    params: Dict[str, Any]
    bytes_in: int
    bytes_out: int

def fc_bytes(IN: int, OUT: int) -> tuple[int, int, int]:
    # activations + weights + bias (int32)
    x = IN
    w = OUT * IN
    b = OUT * 4
    bytes_in = x + w + b
    bytes_out = OUT
    return bytes_in, bytes_out, (w + b)

def dwconv1d_bytes(C: int, L: int, K: int = 3) -> tuple[int, int, int]:
    # input + weights + bias
    x = C * L
    w = C * K
    b = C * 4
    bytes_in = x + w + b
    bytes_out = C * (L - K + 1)
    return bytes_in, bytes_out, (w + b)
