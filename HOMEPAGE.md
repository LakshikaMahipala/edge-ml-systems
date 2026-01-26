ML Hardware & Systems 

Purpose
This repository is my day-by-day learning and build log for becoming strong in ML hardware and ML systems:
measurement, profiling, GPU optimization, FPGA inference kernels, ML compilers (TVM/LLVM), and hardware-aware NAS.
The goal is a public, reproducible portfolio that others can learn from.

Start Here (New Contributors / Beginners)
1) Read docs/glossary.md
2) Read docs/metrics.md (how results are tracked)
3) Follow docs/daily_log.md for progress
4) Start with Project 0: inference_bench/ (benchmarking foundations)

Projects
Project 0 — inference_bench/
- Goal: learn correct latency measurement (warmup, p50/p99), throughput, and accuracy reporting
- Status: active

Project 1 — TinyML-Gesture-TCN/
- Goal: TinyML gesture classification workflow (training → quantization → deployment)
- Status: existing work (will be cleaned later)

Repository Map
- docs/          Notes, glossary, metrics table, daily log
- inference_bench/  Benchmarking mini-project 
- TinyML-Gesture-TCN/  TinyML project work
- .github/workflows/  CI automation (later we will use this for reproducible runs)

Edge build & deployment 
See docs/edge_build_deployment.md for:
- reproducible Docker workflow
- cross-compilation toolchains
- I/O + decode benchmarks
- PyTorch vs ONNX Runtime comparison
- where to record p50/p99 metrics

