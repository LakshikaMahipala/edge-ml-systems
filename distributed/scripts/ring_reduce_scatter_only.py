from __future__ import annotations
import argparse
import numpy as np


def ring_reduce_scatter(vectors: np.ndarray) -> np.ndarray:
    """
    Returns reduced chunks distributed across workers.
    Output shape: (N, chunk)
    """
    N, L = vectors.shape
    assert L % N == 0
    chunk = L // N
    buffers = [[vectors[i, j*chunk:(j+1)*chunk].copy() for j in range(N)] for i in range(N)]

    for s in range(N - 1):
        sends = [None] * N
        recv_idx = [None] * N
        for i in range(N):
            send_chunk_idx = (i - s) % N
            sends[i] = buffers[i][send_chunk_idx].copy()
            recv_idx[i] = (i - s - 1) % N

        for i in range(N):
            src = (i - 1) % N
            idx = recv_idx[i]
            buffers[i][idx] += sends[src]

    # Each worker holds N chunks, but only one of them is the “fully reduced chunk” it owns.
    # Which chunk is fully reduced for worker i? It is recv_idx at last step: (i - (N-1) - 1) = (i - N) = i mod N.
    # In this indexing, worker i ends up owning chunk i.
    owned = np.stack([buffers[i][i] for i in range(N)], axis=0)
    return owned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.N, args.L), dtype=np.float64)

    owned = ring_reduce_scatter(x)
    chunk = args.L // args.N

    # gold: worker i should have sum of chunk i across workers
    gold = np.stack([np.sum(x[:, i*chunk:(i+1)*chunk], axis=0) for i in range(args.N)], axis=0)

    max_err = np.max(np.abs(owned - gold))
    print("Ring reduce-scatter sim")
    print(f"N={args.N}, L={args.L}")
    print(f"max_abs_error={max_err:e}")
    if max_err > 1e-9:
        raise SystemExit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main()
