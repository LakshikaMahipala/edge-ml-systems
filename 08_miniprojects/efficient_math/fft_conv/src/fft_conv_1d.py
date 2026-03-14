from __future__ import annotations
import numpy as np
from utils import next_pow2

def conv1d_full_fft(x: np.ndarray, h: np.ndarray, pad_pow2: bool = True) -> np.ndarray:
    N = x.shape[0]
    K = h.shape[0]
    L = N + K - 1
    P = next_pow2(L) if pad_pow2 else L

    X = np.fft.rfft(x, n=P)
    H = np.fft.rfft(h, n=P)
    Y = X * H
    y = np.fft.irfft(Y, n=P)
    return y[:L].astype(np.float32)
