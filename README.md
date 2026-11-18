# edge-ml-systems

**ML hardware & systems: measuring and optimizing AI inference across GPUs, FPGAs and microcontrollers.**

This repository is my hands-on build log for ML systems work: CUDA kernels, TensorRT deployment, INT8 FPGA kernels in SystemVerilog, hardware-aware neural architecture search, and TinyML on microcontrollers. The common thread is one question: *how fast, how small and how efficient can a model run on a given piece of hardware, and how do we measure that honestly?*

---

## ⭐ Featured work

| Area | What's here | Path | Status |
|---|---|---|---|
| **CUDA kernels** | Vector add, naive GEMM, shared-memory tiled GEMM, convolution-as-GEMM, cuBLAS baseline; CUDA-events timing; scripts to run, parse to CSV and plot | [`02_cuda/cuda_microbench`](02_cuda/cuda_microbench) | 🟡 Code ready · results pending |
| **TensorRT deployment** | PyTorch → ONNX → TensorRT engines (FP32/FP16/INT8 with calibration data), `trtexec` log parsing, FP32-vs-INT8 comparison | [`01_benchmarks/tensorrt_bench`](01_benchmarks/tensorrt_bench) | 🟡 Code ready · results pending |
| **Inference benchmarking harness** | Warm-up, p50/p99 latency, throughput, pre/post-processing breakdown, ONNX Runtime and video-decode benchmarks | [`01_benchmarks/inference_bench`](01_benchmarks/inference_bench) | 🟡 Code ready · results pending |
| **INT8 FPGA kernels (SystemVerilog)** | Fixed-point library (add/mul/saturate/scale-shift), INT8 dot product + requantization, INT8 fully-connected layer (v0 and pipelined v1), INT8 depthwise Conv1D — each with a testbench | [`03_fpga/fpga`](03_fpga/fpga) | 🟢 RTL + testbenches |
| **Binary neural network on FPGA** | XNOR-popcount dot-product core in Verilog with a Python golden reference model | [`03_fpga/fpga_bnn_xnor_popcount`](03_fpga/fpga_bnn_xnor_popcount) | 🟢 RTL + testbench |
| **Hardware-aware NAS** | DARTS and latency-aware DARTS, zero-cost proxies, latency-predictor dataset + models (sklearn / MLP, Spearman rank evaluation), FPGA cost model for ranking architectures | [`04_nas`](04_nas), [`03_fpga/fpga_cost_model`](03_fpga/fpga_cost_model) | 🟡 Code ready · results pending |
| **TinyML gesture recognition** | 5-class gesture classifier from 6-axis IMU data (HGAG-DATA1), trained, quantized and deployed on an Arduino Nano ESP32 | [`99_misc/TinyML-Gesture-TCN`](99_misc/TinyML-Gesture-TCN) | ✅ Measured: **66 % test accuracy** (258 test samples, macro-F1 0.64) |

**Status key:** ✅ measured results in the repo · 🟢 implemented and verified in simulation/testbench · 🟡 implemented, measurements still to be run and committed.
I keep this table honest: a result is only marked ✅ when the raw numbers are committed.

---

## 🗂️ Repository map

| Folder | Contents |
|---|---|
| [`01_benchmarks/`](01_benchmarks) | Inference, transfer and TensorRT benchmarks |
| [`02_cuda/`](02_cuda) | CUDA micro-benchmarks and kernels |
| [`03_fpga/`](03_fpga) | FPGA kernels, fixed-point library, BNN core, overlays, cost models, host (UART) tools |
| [`04_nas/`](04_nas) | NAS (DARTS, RL, one-shot, zero-cost), latency predictors, Pareto analysis, co-design |
| [`05_compilers/`](05_compilers) | TVM Relay, TVM auto-tuning, LLVM IR basics |
| [`06_systems/`](06_systems) | Amdahl analysis, HEFT scheduling, pipeline models, micro-NPU survey |
| [`07_distributed/`](07_distributed) | Distributed and federated learning notes |
| [`08_miniprojects/`](08_miniprojects) | Smaller self-contained experiments (compression on FPGA, NAS ranking, efficient maths) |
| [`docs/`](docs) | Decision guides (TensorRT vs TVM vs FPGA, Jetson deployment), checklists, results policy |
| [`00_home/HOMEPAGE.md`](00_home/HOMEPAGE.md) | Detailed learning path and project notes |

---

## 🛠️ Tools & skills used
CUDA C++ · cuBLAS · TensorRT · ONNX / ONNX Runtime · PyTorch · TVM · SystemVerilog / Verilog · Vivado · Python (NumPy, scikit-learn) · Arduino / ESP32 · CMake · Git

## 🗺️ Next steps
- Run the CUDA and TensorRT benchmarks on an NVIDIA T4 and commit result tables and plots
- Synthesize the INT8 FPGA kernels and report resource use (LUT/DSP/BRAM) and timing
- Validate the latency predictor against measured hardware latency

---

**Author:** Lakshika Padmamali · Embedded Engineer · [GitHub profile](https://github.com/LakshikaMahipala)
