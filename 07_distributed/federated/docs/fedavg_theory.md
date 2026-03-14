Federated Learning + FedAvg 

Setup
We have K clients (phones, edge devices).
Each client k has local dataset D_k.
Goal: train a single global model w without moving raw data off-device.

FedAvg algorithm (core)
Repeat for rounds r = 1..R:
1) Server samples a subset of clients S_r
2) Server sends current global weights w_r to selected clients
3) Each client k runs E local epochs of SGD on its own data:
   w_{r,k} = LocalTrain(w_r, D_k, E)
4) Client sends update (or weights) back to server
5) Server aggregates by weighted average:
   w_{r+1} = sum_k (n_k / sum_j n_j) * w_{r,k}
   where n_k = number of data points on client k

Why FedAvg works
- It approximates centralized SGD when client data is IID and step sizes are small.

Why it’s hard in the real world
1) Non-IID data (clients have different distributions)
- Causes client drift (local updates move in different directions).
- Can slow convergence or reduce final accuracy.

2) Systems constraints
- Clients are slow / intermittent / battery-limited
- Dropouts are normal
- Communication is often the bottleneck, not compute

3) Robustness and security (beyond today)
- poisoned clients, backdoors, etc.

Why this matters for ML hardware
- Training is pushed to constrained devices: compute, memory, energy budgets matter.
- Communication dominates: update compression, quantization, sparsification are key.
- This is the same mindset we use for FPGA I/O bottlenecks.

Today’s repo work
- We implement FedAvg on a tiny linear model with synthetic data.
- We show: IID case vs non-IID case and observe convergence differences.
