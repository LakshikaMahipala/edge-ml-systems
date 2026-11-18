# HW/SW co-design 

Co-design means we do NOT treat the model and hardware separately.

We jointly choose:
- model architecture (blocks, widths, depths, quantization)
- hardware parameters (unroll, tile sizes, buffer allocation, precision datapath)

Because:
A model that is "good" in accuracy may be impossible or inefficient on the target FPGA.
A hardware design that is "fast" may not fit accuracy needs.

Co-design objective:
maximize accuracy subject to latency/energy/area constraints
(or find Pareto set of tradeoffs).
