Mini-Project 1A Report — C++ Preprocessing Speedup vs Python (Week 2 Day 6)

Status
Implemented and documented; execution pending (local run not yet performed).

1) Objective
Quantify how much C++ preprocessing (OpenCV + SIMD path) improves:
- preprocess latency (p50/p99)
- end-to-end latency impact (Amdahl effect)

2) Experiment design (valid speedup rules)
Fixed across both paths:
- model: resnet18 (torchvision pretrained)
- input size: 224x224
- batch: 1
- warmup iterations: 20
- measurement iterations: 100
- device: CPU (initial)
Only changed variable:
- preprocessing implementation (Python vs C++ via pybind11)

3) Scripts
- inference_bench/run_preproc_speedup.py
Path A: Python preprocess -> forward -> postprocess
Path B: C++ preprocess (pybind11) -> forward -> postprocess

4) Metrics to record
For each path:
- preprocess p50/p99
- end-to-end p50/p99
- peak RSS (if available)
Derived:
- preprocess speedup = p50_python / p50_cpp
- end-to-end speedup = p50_e2e_python / p50_e2e_cpp

5) Results (TBD)
Path A (Python):
- preprocess p50/p99: TBD / TBD
- end-to-end p50/p99: TBD / TBD

Path B (C++):
- preprocess p50/p99: TBD / TBD
- end-to-end p50/p99: TBD / TBD

6) Interpretation checklist
- If preprocess speedup is large but end-to-end speedup is small:
  Amdahl bottleneck: inference dominates.
- If end-to-end p99 improves:
  reduced CPU jitter / fewer allocations may be helping tail latency.
- If p99 worsens:
  investigate cache behavior, memory allocation, thread scheduling.

7) Next steps
- Week 2 Day 7 consolidation: document results, generate plots from JSON, tag release.
