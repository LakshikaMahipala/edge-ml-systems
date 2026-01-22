Measurement Methodology 

Purpose

This document defines how we measure performance in this repository.
The goal is reproducibility, correctness, and meaningful comparison across systems.

1) What we measure
   
We report end-to-end latency and (when applicable) component latency:
- Preprocess latency
- Inference latency
- Postprocess latency
- End-to-end latency = preprocess + inference + postprocess + overheads

2) Warmup policy
   
We do warmup iterations before measuring.

Reason:

- caches warm up
- runtime initialization occurs
- allocator behavior stabilizes
Without warmup, early iterations bias results.

3) Percentiles (not averages)
   
We report:

- p50 (median): typical latency
- p99 (tail): worst-case latency under normal jitter
Average latency hides tails and is not acceptable for systems work.

4) Configuration disclosure
   
Every report must state:
- device (CPU/GPU model if known)
- batch size
- input size
- number of warmup iterations
- number of measured iterations
- software versions (Python, torch, OS/platform)

5) Memory reporting
   
We report peak RSS on CPU where possible.
Reason:
- embedded deployments fail due to memory long before compute.

6) Valid comparisons
   
Only compare results when the following are identical:
- model architecture + weights
- input resolution + preprocessing method
- batch size
- measurement methodology
Otherwise, comparisons are invalid.

7) What "end-to-end" means in this repo
   
Unless stated otherwise:
End-to-end includes:
- preprocessing (tensor normalization and formatting)
- forward pass
- postprocessing (top-k)
It does NOT include:
- downloading weights
- first-time model load
- dataset loading from disk (unless explicitly measured)
