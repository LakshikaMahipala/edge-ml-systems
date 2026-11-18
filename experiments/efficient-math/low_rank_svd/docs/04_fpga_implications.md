FPGA implications

Why low-rank is FPGA-friendly
- fewer MACs -> fewer DSPs needed
- smaller matrices -> more likely to fit in BRAM tiles
- can stream A then B sequentially

What to watch
- two-stage pipeline adds latency if not overlapped
- bandwidth: you must fetch/store intermediate vector of size r
- quantization: if INT8, choose scaling for A and B separately
- if I/O dominates (UART), gains may be invisible end-to-end

Design knobs
- choose r aligned with your unroll factor
- fuse scaling into one stage to reduce multipliers
