FPGA sweep dataset schema (v0)

Features
- op_type
- shape: IN/OUT or C/L/K
- unroll factors: UNROLL / UNROLL_C
- II
- f_clk_mhz
- bytes_total
- macs
- cycles_est
- resource_proxy_dsp

Labels
- y_fpga_est_io_ms
- y_fpga_est_compute_ms
- y_fpga_est_total_ms

Why include resource proxies
Later we’ll do co-design / constrained search:
min latency subject to DSP/BRAM limits.
So the dataset must carry resource info.
