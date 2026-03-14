Timing model hook (v0)

Why we need it
Compiler/scheduler decisions require latency estimates.

Timing model components
Total latency per op:
T_op = T_compute + T_memory + T_io + T_overhead

For FPGA kernels today:
- T_io dominates under UART
- We still record compute cycles separately for future faster I/O

v0 timing estimator fields
- op type
- MAC count
- estimated cycles (from kernel design)
- bytes moved (input/output/weights)
- interface bandwidth assumption

Integration point
- A compiler can use timing estimates to choose:
  - tiling factors
  - scheduling order
  - buffer reuse strategy
