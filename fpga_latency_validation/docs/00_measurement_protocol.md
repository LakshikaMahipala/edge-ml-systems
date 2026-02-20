# FPGA latency measurement protocol 

Goal:
Measure a few kernel latencies on FPGA to validate our latency proxy.

We need only 5–10 points.

Protocol:
1) Pick kernel shapes representative of our DARTS ops:
   - conv3-like
   - conv5-like
   - skip-like (copy/no-op)
2) For each kernel:
   - run N times (e.g., N=200)
   - discard warmup (e.g., first 20)
   - report p50 and p99 latency (microseconds)
3) Store results in data/measurements.jsonl

Key rules:
- measure only the kernel (not host preprocessing)
- keep clock frequency fixed and reported
- record bitwidth (INT8/FP16/etc.)
