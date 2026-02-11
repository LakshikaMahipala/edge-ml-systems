# Validation plan for cost model v1 (run later)

1) Choose 10 points per kernel family:
- baseline FC
- low-rank FC
- baseline 3x3 conv
- winograd 3x3 conv (if implemented later)
- 1D conv large kernel vs FFT (optional)

2) Measure in simulation/hardware:
- cycle counts (from RTL counters)
- end-to-end p50/p99 latency (host timer)

3) Fit correction factors:
- II penalty factor
- transform overhead constant terms
- protocol overhead constant

4) Re-evaluate ranking:
- Spearman correlation between predicted and measured
