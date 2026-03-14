Validation plan (v0 -> v1)

Step 1: Synthetic validation (now)
- generate random configs (IN, OUT, C, L)
- ensure estimator outputs are sane and monotonic

Step 2: Measurement validation (later when hardware runs)
For each op:
- run 20 trials with fixed payload sizes
- measure p50/p99 end-to-end
- compare against estimator

Metrics
- absolute error (ms)
- relative error (%)
- rank correlation (Spearman) across configs

Upgrade path to v1
- replace UART-only with interface abstraction (UART/PCIe/Ethernet)
- add overlap model: T_total ≈ max(T_io, T_compute) when streaming overlap exists
- add resource constraints: reject configs that exceed DSP/BRAM budget
