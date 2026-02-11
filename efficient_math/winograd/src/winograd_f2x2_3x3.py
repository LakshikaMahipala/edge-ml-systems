from __future__ import annotations
import numpy as np

# Common Winograd F(2x2,3x3) transform matrices (one standard variant)
B_T = np.array([
    [1,  0, -1,  0],
    [0,  1,  1,  0],
    [0, -1,  1,  0],
    [0,  1,  0, -1],
], dtype=np.float32)

G = np.array([
    [1,   0,   0],
    [0.5, 0.5, 0.5],
    [0.5,-0.5, 0.5],
    [0,   0,   1],
], dtype=np.float32)

A_T = np.array([
    [1,  1,  1,  0],
    [0,  1, -1, -1],
], dtype=np.float32)

def filter_transform(g: np.ndarray) -> np.ndarray:
    # U = G g G^T
    return G @ g @ G.T

def input_transform(d: np.ndarray) -> np.ndarray:
    # V = B^T d B  (B_T is B^T)
    return B_T @ d @ B_T.T

def output_transform(M: np.ndarray) -> np.ndarray:
    # Y = A^T M A  (A_T is A^T)
    return A_T @ M @ A_T.T

def winograd_tile_2x2(d4: np.ndarray, g3: np.ndarray) -> np.ndarray:
    """
    d4: 4x4 input patch
    g3: 3x3 filter
    returns: 2x2 output tile
    """
    U = filter_transform(g3.astype(np.float32))
    V = input_transform(d4.astype(np.float32))
    M = U * V
    Y = output_transform(M)
    return Y
