# ML Hardware & Systems — Public Learning + Build Log

## Purpose
This repository is my day-by-day learning and build log for becoming strong in **ML hardware** and **ML systems**.  
It covers: **latency/throughput measurement**, **profiling**, **CUDA kernel optimization**, **FPGA inference kernels**, **ML compilers (TVM/LLVM)**, and **hardware-aware NAS / co-design**.

The goal is a **public, reproducible portfolio** that others can learn from.

---

## Start Here (Beginners / New Contributors)
If you are new to this repo, follow this exact order:

1. Read `docs/glossary.md` to learn the core terms (latency, throughput, p50/p99, etc.).
2. Read `docs/metrics.md` to understand how results are tracked and reported.
3. Follow `docs/daily_log.md` to see the progress and what was done each day.
4. Start with **Project 0** (benchmarking foundations):  
   `01_benchmarks/inference_bench/`

---

## Projects (What to read first)

### Project 0 — Inference Benchmark Foundations
**Path:** `01_benchmarks/inference_bench/`  
**Goal:** learn correct latency measurement (warmup, p50/p99), throughput, and accuracy reporting.  
**Status:** active (baseline measurement foundation)

**Key outputs you should expect later**
- Preprocess / inference / postprocess breakdown
- End-to-end latency (p50/p99)
- Throughput (items/sec)
- Accuracy metrics (Top-1/Top-5 if applicable)

---

### Project 1 — TinyML Gesture Classification (Legacy Work)
**Path:** `99_misc/TinyML-Gesture-TCN/`  
**Goal:** end-to-end TinyML workflow (training → quantization → deployment).  
**Status:** existing work (will be cleaned and reorganized later)

---

## Repo Map (Where everything lives)

### Benchmarks & Measurement
- `01_benchmarks/`  
  - `inference_bench/` — latency + throughput measurement harness  
  - `transfer_bench/` — transfer overhead and I/O measurement (if present)  
  - `tensorrt_bench/` — TensorRT benchmarking (if present)

### CUDA / GPU Optimization
- `02_cuda/cuda_microbench/` — vector add, naive GEMM, tiled GEMM, cuBLAS comparisons  
- Supporting docs: `docs/summary_cuda.md`, `02_cuda/cuda_microbench/docs/`

### FPGA / Hardware Acceleration
- `03_fpga/` — FPGA kernels, overlays, fixed-point primitives, validation harnesses  
  - `fpga_overlay/` — overlay ISA concepts (Conv/BN/ReLU descriptor idea)  
  - `fpga_latency_validation/` — proxy vs measured validation scaffolds  
  - `host_tools/` — host interface utilities (UART tools etc.)

### NAS / Co-design / Latency Prediction
- `04_nas/` — NAS foundations, proxies, DARTS, supernets, Pareto, co-design loops, predictors

### Compilers / IR
- `05_compilers/` — TVM/Relay/LLVM learning modules

### Systems + Scheduling
- `06_systems/` — transfer budgeting, overlay planning, HEFT scheduling, OPT baseline comparison, FPGA mapping

### Mini-project Reports
- `08_miniprojects/` — weekly mini-project evidence and writeups

---

## Where to record p50/p99 and other metrics (Very Important)
All performance numbers should be stored in **CSV/JSON inside each module’s results folder** and linked from:

- `docs/results_master_index.md` (global index)
- each flagship results table under `flagships/*/results/results_table.csv`

**Recommended standard:**
- Put raw run output JSON under: `<module>/results/`
- Put summary tables under: `<module>/results/summary.csv`
- Update the master index: `docs/results_master_index.md`

Example (Inference Bench):
- `01_benchmarks/inference_bench/results/bench_summary.csv`
- `01_benchmarks/inference_bench/results/*.json`

---

## Edge Build & Deployment Notes
For deployment workflow guidance (Docker, cross-compilation, edge packaging), see:
- `docs/edge_build_deployment.md`

---

## CI / Automation (Later)
CI automation will live under:
- `.github/workflows/`

Later, the goal is to support reproducible runs and auto-updated result tables.
