from __future__ import annotations
import numpy as np

def _split(M: np.ndarray):
    n = M.shape[0]
    k = n // 2
    return M[:k, :k], M[:k, k:], M[k:, :k], M[k:, k:]

def strassen(A: np.ndarray, B: np.ndarray, leaf_size: int = 64) -> np.ndarray:
    """
    Strassen matrix multiplication for square matrices A,B.
    Uses recursion down to leaf_size then switches to np.dot.
    Assumes A,B are (n,n) with n power-of-two. (Use padding externally if needed.)
    """
    n = A.shape[0]
    if n <= leaf_size:
        return A @ B

    A11, A12, A21, A22 = _split(A)
    B11, B12, B21, B22 = _split(B)

    M1 = strassen(A11 + A22, B11 + B22, leaf_size)
    M2 = strassen(A21 + A22, B11,        leaf_size)
    M3 = strassen(A11,        B12 - B22, leaf_size)
    M4 = strassen(A22,        B21 - B11, leaf_size)
    M5 = strassen(A11 + A12,  B22,       leaf_size)
    M6 = strassen(A21 - A11,  B11 + B12, leaf_size)
    M7 = strassen(A12 - A22,  B21 + B22, leaf_size)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    top = np.concatenate([C11, C12], axis=1)
    bot = np.concatenate([C21, C22], axis=1)
    return np.concatenate([top, bot], axis=0)
