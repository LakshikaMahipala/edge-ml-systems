from __future__ import annotations
import numpy as np

def conv2d_valid_single_channel(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    H, W = d.shape
    KH, KW = g.shape
    OH, OW = H - KH + 1, W - KW + 1
    out = np.zeros((OH, OW), dtype=np.float32)
    for y in range(OH):
        for x in range(OW):
            out[y, x] = float(np.sum(d[y:y+KH, x:x+KW] * g))
    return out
