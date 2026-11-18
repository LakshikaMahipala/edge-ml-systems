Run-later validation (when FPGA sim/hw is available)

For selected points:
- choose 5–10 sweep points covering small to large unroll
- measure end-to-end latency p50/p99 in simulation or hardware
- compare against estimator
- fit a correction factor for II and overhead

Metrics
- absolute error (ms)
- relative error (%)
- Spearman rank correlation across sweep points
