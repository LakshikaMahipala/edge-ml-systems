Tiny CNN cost model (MACs -> cycles, rough)

Goal
We cannot measure on MCU today, so we estimate.

MACs
For conv:
MACs = H_out * W_out * C_out * (K_h*K_w*C_in)

For FC:
MACs = C_in * C_out

Cycles estimate (very rough)
cycles ≈ MACs / (MACs_per_cycle_effective)

On Cortex-M:
- MACs_per_cycle is not 1 in practice because of memory and quantization overhead.
- CMSIS-NN improves effective throughput.

We will use a simplified proxy:
cycles ≈ MACs * alpha
where alpha is a constant depending on assumed efficiency.

This proxy is not to claim real performance.
It is to compare model variants consistently until we can benchmark on real hardware.
