from __future__ import annotations
import numpy as np

def conv2d_valid_single_channel(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    d: HxW input
    g: 3x3 filter
    returns: (H-2)x(W-2)
    """
    H, W = d.shape
    out = np.zeros((H - 2, W - 2), dtype=d.dtype)
    for y in range(H - 2):
        for x in range(W - 2):
            patch = d[y:y+3, x:x+3]
            out[y, x] = np.sum(patch * g)
    return out
