from __future__ import annotations
import numpy as np

def conv2d_full_fft(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    H, W = d.shape
    KH, KW = g.shape
    FH, FW = H + KH - 1, W + KW - 1

    D = np.fft.rfft2(d, s=(FH, FW))
    G = np.fft.rfft2(g, s=(FH, FW))
    Y = D * G
    y = np.fft.irfft2(Y, s=(FH, FW))
    return y.astype(np.float32)

def conv2d_valid_fft(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """'Valid' cross-correlation (the CNN convention) computed with FFTs.

    The FFT computes true convolution, which flips the kernel. CNN layers (and
    conv2d_valid_single_channel) do not flip it, so we flip g first.
    """
    full = conv2d_full_fft(d, g[::-1, ::-1])
    KH, KW = g.shape
    # valid region starts at (KH-1, KW-1)
    return full[KH-1: full.shape[0]-(KH-1), KW-1: full.shape[1]-(KW-1)]
