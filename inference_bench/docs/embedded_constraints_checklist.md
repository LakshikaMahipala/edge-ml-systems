Embedded Constraints Checklist 

Why this exists
In embedded/edge ML, success is not “the model runs.” Success is:
- It fits (RAM/flash/storage)
- It meets latency requirements (especially p99)
- It sustains throughput without tail latency exploding
- It stays within power/thermal limits
- It is reproducible (same results across machines)

1) Memory constraints
- Model weights: parameters * bytes_per_param (FP32=4, FP16=2, INT8=1)
- Activations: often dominate peak memory for CNN/ViT; depends on batch and layer shapes
- Peak RSS matters on CPU; GPU has separate allocator behavior
Hidden knowledge:
- Batch size increases throughput but can blow up activation memory.
- “It runs once” is not enough: memory fragmentation can appear over time.

2) Compute constraints
- FLOPs/MACs are not latency. Real latency depends on:
  - kernel implementations
  - vectorization
  - cache locality
  - operator fusion
Hidden knowledge:
- Many models are bandwidth-bound, not compute-bound, especially on CPU.

3) Bandwidth / data movement
- Preprocess (decode/resize) can dominate end-to-end.
- CPU↔GPU transfer (PCIe) can dominate if not pipelined.
Hidden knowledge:
- Optimizing compute while ignoring input pipeline often yields near-zero end-to-end gains (Amdahl).

4) Tail latency (p99) and stability
- p50 is “typical.”
- p99 is what users feel when the system is under load, or OS jitter occurs.
Hidden knowledge:
- GC, background tasks, CPU frequency scaling, and cache misses tend to inflate p99.

5) Power/thermal (edge reality)
- Sustained workload causes throttling.
Hidden knowledge:
- Your first benchmark run may be faster than steady-state due to boosting.

6) Toolchain constraints
- Dependencies should be minimal.
- Cross-compilation may restrict Python usage.
Hidden knowledge:
- “Works on my machine” usually means hidden dynamic library dependencies.

Day 5 policy in this repo
Every benchmark must report:
- p50 and p99 end-to-end latency
- batch size and input size
- device + system info
- memory peak (where possible)
