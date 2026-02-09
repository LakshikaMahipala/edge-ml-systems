from __future__ import annotations
import numpy as np


def sat8(v: np.ndarray) -> np.ndarray:
    return np.clip(v, -128, 127).astype(np.int8)


def rshift_round(v: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return v.astype(np.int32)
    return ((v.astype(np.int32) + (1 << (shift - 1))) >> shift).astype(np.int32)


def dwconv1d_int8(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """
    x: int8 [C, L]
    w: int8 [C, K=3]
    b: int32 [C]
    return: int8 [C, L-2]
    """
    assert x.dtype == np.int8 and w.dtype == np.int8 and b.dtype == np.int32
    C, L = x.shape
    K = w.shape[1]
    assert K == 3
    out = np.zeros((C, L - K + 1), dtype=np.int8)

    for c in range(C):
        for t in range(L - K + 1):
            acc = np.int32(0)
            for k in range(K):
                acc += np.int32(x[c, t + k]) * np.int32(w[c, k])
            acc += b[c]
            y32 = rshift_round(np.array([acc], dtype=np.int32), shift)[0]
            out[c, t] = sat8(np.array([y32], dtype=np.int32))[0]
    return out


def relu_int8(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0).astype(np.int8)


def mobilenet_block_v0(x: np.ndarray, w_dw: np.ndarray, b_dw: np.ndarray, shift: int) -> np.ndarray:
    y1 = dwconv1d_int8(x, w_dw, b_dw, shift=shift)
    y2 = relu_int8(y1)
    return y2


def make_deterministic_weights(C: int = 4, K: int = 3, shift: int = 7):
    # match your RTL pattern from dwconv1d_int8_v0
    w = np.zeros((C, K), dtype=np.int8)
    b = np.zeros((C,), dtype=np.int32)
    for c in range(C):
        b[c] = np.int32((c - 1) << shift)
        for k in range(K):
            if k == 0:
                w[c, k] = np.int8(-1 * (c + 1))
            elif k == 1:
                w[c, k] = np.int8(2 * (c + 1))
            else:
                w[c, k] = np.int8(-1 * (c + 1))
    return w, b


def make_deterministic_input(C: int = 4, L: int = 16):
    x = np.zeros((C, L), dtype=np.int8)
    for c in range(C):
        for i in range(L):
            x[c, i] = np.int8(((c * 7 + i * 3) % 31) - 15)
    return x


if __name__ == "__main__":
    C, L, shift = 4, 16, 7
    x = make_deterministic_input(C, L)
    w, b = make_deterministic_weights(C, 3, shift)

    y = mobilenet_block_v0(x, w, b, shift)
    print("Input x:\n", x)
    print("DW weights:\n", w)
    print("Bias:\n", b)
    print("Output y (DW+ReLU):\n", y)
