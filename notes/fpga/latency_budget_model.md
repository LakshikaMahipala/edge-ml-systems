Latency Budget Model (for accelerator thinking)

Why this exists
When you “accelerate inference”, you rarely accelerate the whole pipeline.
Most systems have:
- preprocess (CPU)
- copy to accelerator
- accelerator compute
- copy back
- postprocess (CPU)

Total latency:
T_total = T_pre + T_infer + T_post
If offloading part of inference:
T_total = T_pre + T_copy_in + T_accel_compute + T_copy_out + T_post

The trap
Even if FPGA/GPU compute is extremely fast, copies can dominate.

Speedup bound (Amdahl-style)
If fraction f of time is non-accelerated, max speedup is:
S_max = 1 / f

Practical example (UART FPGA)
UART might be ~1 Mbps to a few Mbps effective.
If your tensor is large, copy time dominates.
Therefore early FPGA demos must:
- use small models first (MLP, tiny conv)
- reduce I/O (quantize, pack, stream)
- pipeline I/O and compute

This model is implemented in tools/latency_budget.py
