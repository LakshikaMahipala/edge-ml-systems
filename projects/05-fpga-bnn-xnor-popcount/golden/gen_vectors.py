"""Random test vectors for tb_bnn_dot: packed words and expected dot products."""
import argparse
from pathlib import Path

import numpy as np

from bnn_ref import dot_xnor_popcount

N_BITS, WORD_W = 256, 32
N_WORDS = N_BITS // WORD_W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vectors")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    a = rng.integers(0, 2**32, (args.tests, N_WORDS), dtype=np.uint64)
    w = rng.integers(0, 2**32, (args.tests, N_WORDS), dtype=np.uint64)
    a[0], w[0] = 0xFFFFFFFF, 0xFFFFFFFF  # all agree    -> +256
    a[1], w[1] = 0xFFFFFFFF, 0           # all disagree -> -256
    acc = [dot_xnor_popcount(list(map(int, ra)), list(map(int, rw))) for ra, rw in zip(a, w)]

    for name, vals in [("bnn_a.hex", a.ravel()), ("bnn_w.hex", w.ravel())]:
        (out / name).write_text("".join(f"{int(v):08x}\n" for v in vals))
    (out / "bnn_acc.hex").write_text("".join(f"{v & 0xFFFFFFFF:08x}\n" for v in acc))
    (out / "config.svh").write_text(
        f"`define N_TESTS {args.tests}\n`define N_BITS {N_BITS}\n`define WORD_W {WORD_W}\n")
    print(f"wrote {args.tests} BNN dot-product vectors to {out}/")


if __name__ == "__main__":
    main()
