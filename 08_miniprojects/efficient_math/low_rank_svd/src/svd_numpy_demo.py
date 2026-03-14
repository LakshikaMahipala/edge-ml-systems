from __future__ import annotations
import argparse
import numpy as np

def low_rank_approx(W: np.ndarray, r: int):
    # W: (in, out)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U_r = U[:, :r]         # (in, r)
    S_r = S[:r]            # (r,)
    Vt_r = Vt[:r, :]       # (r, out)

    # Factorization: W_r = (U_r * S_r) @ Vt_r
    A = U_r * S_r          # broadcasting => (in, r)
    B = Vt_r               # (r, out)
    return A, B

def energy_curve(S: np.ndarray):
    e = (S ** 2)
    cum = np.cumsum(e) / np.sum(e)
    return cum

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dim", type=int, default=512)
    ap.add_argument("--out_dim", type=int, default=512)
    ap.add_argument("--r", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    W = rng.standard_normal((args.in_dim, args.out_dim)).astype(np.float32)
    x = rng.standard_normal((1, args.in_dim)).astype(np.float32)

    y_ref = x @ W

    A, B = low_rank_approx(W, args.r)
    y_lr = (x @ A) @ B

    err = float(np.max(np.abs(y_ref - y_lr)))
    rel = float(err / (np.max(np.abs(y_ref)) + 1e-9))

    # energy
    _, S, _ = np.linalg.svd(W, full_matrices=False)
    cum = energy_curve(S)
    print("rank r:", args.r)
    print("energy@r:", float(cum[args.r - 1]))
    print("max_abs_err:", err)
    print("rel_err:", rel)

if __name__ == "__main__":
    main()
