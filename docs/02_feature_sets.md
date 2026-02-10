# Feature sets

## Common features
- bytes_total
- macs
- arith_intensity = macs / bytes_total
- log1p(macs), log1p(bytes_total)
- f_clk_mhz, interface baud (FPGA)

## CPU/GPU schedule-like features (tiling)
- tile_co, tile_ci, tile_y, tile_x, vec, unroll

## FPGA schedule-like features
- UNROLL (FC) or UNROLL_C (DWConv)
- II

Why include schedule params directly?
Because latency differences come from implementation choices even with same shape.
