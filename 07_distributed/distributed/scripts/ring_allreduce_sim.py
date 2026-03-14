from __future__ import annotations
import argparse
import numpy as np


def ring_allreduce(vectors: np.ndarray) -> np.ndarray:
    """
    vectors: shape (N, L) float32/float64
    returns: shape (N, L) where each row is the allreduced sum
    Implements ring reduce-scatter + allgather.
    """
    N, L = vectors.shape
    assert L % N == 0, "For this toy sim, L must be divisible by N"
    chunk = L // N

    # Split into chunks: buffers[i][j] = chunk j owned by worker i
    buffers = [[vectors[i, j*chunk:(j+1)*chunk].copy() for j in range(N)] for i in range(N)]

    # -------------------------
    # Phase 1: Reduce-Scatter
    # -------------------------
    # Each worker ends with one reduced chunk (different chunk per worker)
    # We'll implement the standard pattern:
    # step s: worker i sends chunk (i - s) to i+1 and receives chunk (i - s - 1) from i-1, accumulates it.
    for s in range(N - 1):
        sends = [None] * N
        recv_idx = [None] * N

        for i in range(N):
            send_chunk_idx = (i - s) % N
            sends[i] = buffers[i][send_chunk_idx].copy()
            recv_idx[i] = (i - s - 1) % N

        # deliver + accumulate
        for i in range(N):
            src = (i - 1) % N
            idx = recv_idx[i]
            buffers[i][idx] += sends[src]

    # After reduce-scatter, worker i "owns" chunk (i - (N-1)) == (i+1)?? depends on indexing;
    # But correctness doesn't require naming; we will proceed to allgather using the same movement.

    # -------------------------
    # Phase 2: Allgather
    # -------------------------
    # step s: worker i sends chunk (i - s - 1) to i+1 and receives chunk (i - s - 2) from i-1
    for s in range(N - 1):
        sends = [None] * N
        send_idx = [None] * N
        recv_i = [None] * N

        for i in range(N):
            send_idx[i] = (i - s - 1) % N
            sends[i] = buffers[i][send_idx[i]].copy()
            recv_i[i] = (i - s - 2) % N

        for i in range(N):
            src = (i - 1) % N
            idx = recv_i[i]
            buffers[i][idx] = sends[src]

    # Reassemble
    out = np.zeros((N, L), dtype=vectors.dtype)
    for i in range(N):
        out[i] = np.concatenate(buffers[i], axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--avg", action="store_true", help="return average instead of sum")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.N, args.L), dtype=np.float64)

    y = ring_allreduce(x)
    gold = np.sum(x, axis=0, keepdims=True).repeat(args.N, axis=0)

    if args.avg:
        y = y / args.N
        gold = gold / args.N

    max_err = np.max(np.abs(y - gold))
    print("Ring allreduce sim")
    print(f"N={args.N}, L={args.L}, avg={args.avg}")
    print(f"max_abs_error={max_err:e}")

    # Fail fast if wrong
    if max_err > 1e-9:
        raise SystemExit("FAIL: incorrect allreduce")
    print("PASS")


if __name__ == "__main__":
    main()
