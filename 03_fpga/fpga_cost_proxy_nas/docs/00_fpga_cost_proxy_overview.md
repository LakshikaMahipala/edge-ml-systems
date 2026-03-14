# FPGA cost proxy in NAS 

Goal:
Given an architecture encoding, estimate "hardware pain" without building RTL.

Outputs:
- cycles_proxy: how many cycles per inference for main ops
- lut_proxy: logic usage pressure (relative)
- bram_proxy: memory footprint pressure (relative)

Use:
score_hw = acc_proxy - λ * normalized_cost

This is the first step toward real co-design.
Later we replace proxy with measured kernel latency points.
