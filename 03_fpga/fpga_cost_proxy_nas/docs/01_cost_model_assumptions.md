# Cost model assumptions (explicit)

We assume a simple FPGA datapath style:
- streaming conv / mbconv blocks
- fixed-point INT8
- vectorized MAC parallelism = P (a tunable parameter)

Assumptions:
- latency ~ total_ops / P  (cycles proxy)
- LUT cost increases with:
  - parallelism P
  - channel width
  - popcount/adder tree complexity in BNN-like ops
- BRAM cost increases with:
  - activation buffering (feature map size)
  - weight storage for conv/pointwise

These are *not* real synthesis numbers.
They are ranking proxies.
