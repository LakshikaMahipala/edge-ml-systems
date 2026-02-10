# End-to-end bottleneck examples 

## Example 1: UART kills compute speedups (our current FPGA regime)
If UART is 115200 baud:
- bytes/sec ≈ 115200/10 = 11520 B/s
If your payload is 100 KB:
- T_io ≈ 100000 / 11520 ≈ 8.68 sec

Even if FPGA compute becomes 1000× faster, total time barely changes.
**Conclusion:** you must treat I/O as first-class in HW acceleration claims.

## Example 2: GPU kernel speedups vs preprocessing
If preprocessing takes 12 ms and inference takes 4 ms:
- total = 16 ms
Even making inference 2× faster saves only 2 ms.
Amdahl says the serial part caps speedup.

## Example 3: batching tradeoff
Batching improves throughput, often hurts per-item latency.
- TensorRT may prefer larger batch for GPU occupancy
- real-time systems may require batch=1

## What this means for our roadmap
- We use cost models that explicitly include I/O + compute.
- We report p50/p99 latency, not only mean.
