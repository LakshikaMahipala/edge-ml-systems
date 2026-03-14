# Hardware-aware NAS recipe (DARTS + FPGA proxy)

Step 1 — Choose target metric
- latency p50 (interactive)
- latency p99 (real-time guarantees)
Pick one.

Step 2 — Build a latency estimator ladder
L0: FLOPs proxy
L1: layerwise latency table
L2: FPGA proxy cycles (Week 13 model)
L3: measured FPGA points for calibration

Step 3 — Make latency differentiable for DARTS
Use expected latency:
E[lat] = Σ p_k * lat(op_k)

Step 4 — Tune λ
- start small (λ=0.05–0.2 range)
- observe tradeoff curve: val_loss vs latency_total
- choose λ that shifts ops without destroying accuracy

Step 5 — Validate estimator with measurements
- measure 5–10 kernel points
- check Spearman correlation
- if Spearman is low → estimator must be fixed before using in NAS

Step 6 — Final selection
Even after DARTS search:
- rebuild discrete network
- retrain from scratch
- measure real latency
Only then you can claim “hardware-aware improvement”.
