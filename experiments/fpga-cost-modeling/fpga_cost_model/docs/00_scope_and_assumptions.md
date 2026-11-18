FPGA latency estimator v0 — scope and assumptions

Goal
Estimate end-to-end latency for our FPGA backend:
T_total = T_io + T_compute + T_host_overhead

What this v0 supports
- INT8_FC
- INT8_DWCONV1D_K3
- MobileNet-like block v0 = DWConv1D + ReLU (ReLU is negligible compute, but may have I/O effects)

Key assumptions (v0)
1) UART transport dominates I/O
- baud default: 115200
- effective bytes/s ≈ baud/10

2) Compute cycle models are simple
- FC cycles ≈ IN (+ overhead)
- DWConv cycles ≈ C*(L-K+1)*K

3) We assume fixed FPGA clock f_clk (default 100 MHz)

4) Host overhead is either:
- provided as a constant ms (default 0)
or
- inferred later from measured runs

Output format
- prints breakdown and a JSON summary

Non-goals (v0)
- accurate DSP packing/BRAM modeling
- overlap modeling (DMA + compute overlap)
- PCIe model (later)
