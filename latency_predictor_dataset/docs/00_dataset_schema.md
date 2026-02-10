Latency predictor dataset schema 

Goal
Create a dataset of (layer configuration -> latency).

We store one row per layer/op instance.

Core columns
- op_type: categorical
- shape parameters: integers (op-specific)
- quant: SHIFT (int) or placeholder
- bytes_in, bytes_out (int)
- macs (int)
- cycles_est (int)  (for FPGA-like kernels)
- interface: uart/pcie/etc (string; v0 only uart)

Labels (targets)
- y_fpga_est_total_ms (float)
- y_fpga_est_io_ms (float)
- y_fpga_est_compute_ms (float)
- y_cpu_measured_ms (float, blank now)
- y_gpu_measured_ms (float, blank now)

Metadata
- seed
- notes

Why separate I/O and compute?
Because on real systems, bottlenecks move.
If you only learn total latency, you lose interpretability.
