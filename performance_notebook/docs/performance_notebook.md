Performance Notebook 

Purpose
This document is the single source of truth for performance analysis in this repo.
It connects:
- stage breakdown timing (pre/infer/post)
- copy/I/O overhead
- pipeline bottleneck + queueing effects (p50/p99)
- Amdahl speedup bounds
- FPGA offload positioning

What we measure (minimum)
For each pipeline/backend, record:
- preprocess p50/p99 (ms)
- inference p50/p99 (ms)
- postprocess p50/p99 (ms)
- copies/I-O p50/p99 (ms) where applicable
- end-to-end p50/p99 (ms)
- throughput estimate (items/s or FPS)

Canonical backends
1) PyTorch (CPU baseline)
2) ORT (CPU)
3) TensorRT (FP32/FP16/INT8, DLA optional)
4) FPGA kernel (UART now; PCIe later)

The rule: I/O is first-class
Any FPGA “speedup claim” is invalid unless we include I/O.
We report:
T_total = T_pre + T_infer + T_post + T_io

Amdahl use
We treat inference as the accelerated part and compute:
- p = T_infer / T_total
- max possible speedup if inference were infinitely fast
- what-if speedup for TRT/FPGA factors

Queueing use (video/demo pipelines)
If arrivals approach capacity, p99 grows due to queueing.
We use pipeline_model/scripts/pipeline_sim.py to show the expected p99 inflation.

Where results live
- Raw run JSON:
  performance_notebook/results/*.json
- Summary tables:
  performance_notebook/docs/results_template.md
- Week 7 summary:
  docs/week7_consolidation.md
