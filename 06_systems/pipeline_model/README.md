Pipeline Model 

Goal
- Explain and model the inference pipeline as a queueing system.

Scripts
1) scripts/pipeline_budget.py
- Deterministic latency + throughput bottleneck math

2) scripts/pipeline_sim.py
- Queue simulation (Poisson arrivals) to estimate p50/p99

Run later (example)
python scripts/pipeline_budget.py --t_pre_ms 3 --t_inf_ms 12 --t_post_ms 2 --copies_ms 1
python scripts/pipeline_sim.py --t_pre_ms 3 --t_inf_ms 12 --t_post_ms 2 --fps 30
