"""Bit-exact reference for the XNOR-popcount binary dot product.

Binary neural networks restrict weights and activations to {-1, +1} and store
each value as one bit (1 -> +1, 0 -> -1). For two +-1 vectors of length n:

    dot(a, w) = (#positions that agree) - (#positions that differ)
              = 2 * popcount(XNOR(a_bits, w_bits)) - n

so a 32-element multiply-accumulate becomes one XNOR, one popcount and one
add: a few LUTs on an FPGA instead of 32 multipliers.
"""
from __future__ import annotations

import numpy as np

WORD_W = 32
MASK = (1 << WORD_W) - 1


def popcount32(x: int) -> int:
    return bin(x & MASK).count("1")


def xnor32(a: int, w: int) -> int:
    return ~(a ^ w) & MASK


def dot_xnor_popcount(words_a: list[int], words_w: list[int]) -> int:
    """What the hardware computes, word by word."""
    assert len(words_a) == len(words_w)
    return sum(2 * popcount32(xnor32(a, w)) - WORD_W for a, w in zip(words_a, words_w))


def pack_pm1(v: np.ndarray) -> list[int]:
    """Pack a +-1 vector (length multiple of 32) into 32-bit words, element 0 = bit 0."""
    bits = (np.asarray(v) > 0).astype(np.uint64).reshape(-1, WORD_W)
    return [int((row << np.arange(WORD_W, dtype=np.uint64)).sum()) for row in bits]


def dot_pm1(a: np.ndarray, w: np.ndarray) -> int:
    """Plain integer dot product of two +-1 vectors (the maths the trick replaces)."""
    return int(np.dot(np.asarray(a, dtype=np.int64), np.asarray(w, dtype=np.int64)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.choice([-1, 1], 64)
    w = rng.choice([-1, 1], 64)
    print("plain +-1 dot product :", dot_pm1(a, w))
    print("XNOR-popcount version :", dot_xnor_popcount(pack_pm1(a), pack_pm1(w)))
