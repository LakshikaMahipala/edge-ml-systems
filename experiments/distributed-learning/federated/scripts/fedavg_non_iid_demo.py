from __future__ import annotations
import argparse
import numpy as np
from fedavg_sim import fedavg, mse


def make_non_iid_clients(K: int, n_per: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    w_true, b_true = 2.0, -1.0
    clients = []

    # each client has a different x-range (biased)
    centers = np.linspace(-2, 2, K)
    for k in range(K):
        c = centers[k]
        x = rng.uniform(c - 0.5, c + 0.5, size=(n_per, 1))
        noise = rng.normal(0, 0.2, size=(n_per, 1))
        y = w_true * x + b_true + noise
        clients.append((x, y))
    return clients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_per", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--client_frac", type=float, default=0.5)
    ap.add_argument("--local_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clients = make_non_iid_clients(args.K, args.n_per, seed=args.seed)
    wb, hist = fedavg(
        clients,
        rounds=args.rounds,
        client_frac=args.client_frac,
        local_steps=args.local_steps,
        lr=args.lr,
        seed=args.seed,
        dropout_p=0.0,
    )

    Xg = np.vstack([c[0] for c in clients])
    Yg = np.vstack([c[1] for c in clients])
    loss = mse(wb, Xg, Yg)

    print("FedAvg Non-IID sim")
    print(f"Final wb: w={wb[0]:.4f}, b={wb[1]:.4f}")
    print(f"Final global loss: {loss:.6f}")
    print("Last 5 rounds: [r, w, b, loss, num_clients_used]")
    print(hist[-5:])

    print("\nInterpretation:")
    print("- Compare this loss and convergence to IID case.")
    print("- Non-IID usually converges slower / worse for same hyperparameters.")
    print("- Reducing local_steps often helps stability under non-IID.")

if __name__ == "__main__":
    main()
