# Objectives and constraints

Objectives:
- Accuracy proxy: acc_proxy(arch)  (later: real accuracy)
- Latency proxy: lat_proxy(arch, hw)
- Energy proxy: energy_proxy(arch, hw) ~ lat_proxy * power_proxy

Constraints (examples):
- latency <= budget_ms
- BRAM proxy <= budget
- LUT proxy <= budget

We will primarily use Pareto evaluation (Week 15 Day 1) instead of a single scalar score.
