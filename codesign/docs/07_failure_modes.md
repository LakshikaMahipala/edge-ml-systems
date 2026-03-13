# Co-design failure modes

1) Proxy mismatch
- chosen design looks good in proxy, bad in real measurement

2) Search space mismatch
- missing a crucial knob (e.g., memory layout) causes false optimum

3) Unfair comparisons
- some candidates require different toolchain settings

4) Over-weighting one objective
- latency-only search collapses accuracy

Mitigation:
- periodic calibration (aging evaluation)
- Pareto reporting (don’t hide tradeoffs)
- keep evidence rules (Week 12 honesty rules style)
