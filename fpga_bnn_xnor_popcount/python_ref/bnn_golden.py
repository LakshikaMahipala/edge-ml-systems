from __future__ import annotations

def popcount32(x: int) -> int:
    return bin(x & 0xFFFFFFFF).count("1")

def xnor32(a: int, w: int) -> int:
    return (~(a ^ w)) & 0xFFFFFFFF

def dot_acc(words_a: list[int], words_w: list[int], word_w: int = 32) -> int:
    assert len(words_a) == len(words_w)
    acc = 0
    for a, w in zip(words_a, words_w):
        m = popcount32(xnor32(a, w))
        acc += 2 * m - word_w
    return int(acc)

def demo():
    a0 = 0xFFFFFFFF
    w0 = 0xFFFFFFFF
    a1 = 0xFFFFFFFF
    w1 = 0x00000000
    acc = dot_acc([a0, a1], [w0, w1])
    print("expected acc =", acc)

if __name__ == "__main__":
    demo()
