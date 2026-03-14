# Pareto dominance

We define objectives with direction:
- accuracy: maximize
- latency: minimize
- energy: minimize

Model A dominates model B if:
- accuracy_A >= accuracy_B
- latency_A <= latency_B
- energy_A <= energy_B
and at least one strict inequality holds.

Dominated models should be removed before making decisions.
