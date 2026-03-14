from __future__ import annotations
import argparse
import numpy as np


def make_iid_clients(K: int, n_per: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    # True model
    w_true, b_true = 2.0, -1.0
    clients = []
    for _ in range(K):
        x = rng.uniform(-2, 2, size=(n_per, 1))
        noise = rng.normal(0, 0.2, size=(n_per, 1))
        y = w_true * x + b_true + noise
        clients.append((x, y))
    return clients


def local_sgd(wb: np.ndarray, x: np.ndarray, y: np.ndarray, lr: float, steps: int):
    # wb = [w, b]
    w, b = wb[0], wb[1]
    n = x.shape[0]
    for _ in range(steps):
        # full-batch gradient for simplicity (deterministic)
        yhat = w * x + b
        err = yhat - y
        dw = float((2.0 / n) * np.sum(err * x))
        db = float((2.0 / n) * np.sum(err))
        w -= lr * dw
        b -= lr * db
    return np.array([w, b], dtype=np.float64)


def mse(wb: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    w, b = wb[0], wb[1]
    yhat = w * x + b
    return float(np.mean((yhat - y) ** 2))


def fedavg(
    clients,
    rounds: int,
    client_frac: float,
    local_steps: int,
    lr: float,
    seed: int = 0,
    dropout_p: float = 0.0,
):
    rng = np.random.default_rng(seed)
    K = len(clients)
    wb = np.array([0.0, 0.0], dtype=np.float64)

    # global eval set = union of all data
    Xg = np.vstack([c[0] for c in clients])
    Yg = np.vstack([c[1] for c in clients])

    hist = []
    for r in range(rounds):
        m = max(1, int(client_frac * K))
        selected = rng.choice(K, size=m, replace=False)

        updates = []
        weights = []

        for k in selected:
            # simulate dropout
            if dropout_p > 0 and rng.random() < dropout_p:
                continue
            x, y = clients[k]
            wb_k = local_sgd(wb.copy(), x, y, lr=lr, steps=local_steps)
            updates.append(wb_k)
            weights.append(x.shape[0])

        if not updates:
            # if all dropped, keep old wb
            loss = mse(wb, Xg, Yg)
            hist.append((r, wb[0], wb[1], loss, 0))
            continue

        W = np.array(weights, dtype=np.float64)
        W = W / np.sum(W)
        wb = np.sum(np.stack(updates, axis=0) * W[:, None], axis=0)

        loss = mse(wb, Xg, Yg)
        hist.append((r, wb[0], wb[1], loss, len(updates)))

    return wb, np.array(hist, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--n_per", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--client_frac", type=float, default=0.5)
    ap.add_argument("--local_steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--dropout_p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clients = make_iid_clients(args.K, args.n_per, seed=args.seed)
    wb, hist = fedavg(
        clients,
        rounds=args.rounds,
        client_frac=args.client_frac,
        local_steps=args.local_steps,
        lr=args.lr,
        seed=args.seed,
        dropout_p=args.dropout_p,
    )

    print("FedAvg IID sim")
    print(f"Final wb: w={wb[0]:.4f}, b={wb[1]:.4f}")
    print(f"Final loss: {hist[-1,3]:.6f}")
    print("Last 5 rounds: [r, w, b, loss, num_clients_used]")
    print(hist[-5:])

if __name__ == "__main__":
    main()
