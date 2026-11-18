# edge-ml-systems

**Making neural networks run fast and small on real hardware: GPUs, FPGAs and microcontrollers.**

[![CI](https://github.com/LakshikaMahipala/edge-ml-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/LakshikaMahipala/edge-ml-systems/actions/workflows/ci.yml)

This repository collects my work on ML systems and hardware-aware AI: hand-written CUDA kernels, TensorRT INT8 deployment, INT8 and binary neural-network hardware in SystemVerilog, and TinyML on an ESP32. Every project states what was measured and how to reproduce it, and separates *verified* from *pending* results.

## ⭐ Projects

| # | Project | What it shows | Stack | Status |
|---|---|---|---|---|
| 01 | [**CUDA SGEMM optimization**](projects/01-cuda-gemm-optimization) | Naive → shared-memory tiling → 2D register blocking, each checked against cuBLAS, with an arithmetic-intensity analysis | CUDA C++, cuBLAS | 🟡 code complete · T4 results pending |
| 02 | [**TensorRT FP32 / FP16 / INT8**](projects/02-tensorrt-int8-quantization) | Accuracy *and* latency of INT8 post-training quantization on a T4, with entropy calibration on a real labelled dataset | TensorRT 10, PyTorch, ONNX | 🟡 pipeline complete · T4 results pending |
| 03 | [**Inference benchmark harness**](projects/03-inference-benchmark-harness) | p50/p99 latency, pre/post-processing breakdown, transfer cost and queueing effects | PyTorch, ONNX Runtime, C++/OpenCV | 🟡 CPU tools verified |
| 04 | [**INT8 NN kernels in SystemVerilog**](projects/04-fpga-int8-kernels) | FC (parallel and sequential) and depthwise-conv layers, bit-exact against a Python golden model | SystemVerilog, Python | 🟢 verified in CI |
| 05 | [**Binary NN XNOR-popcount core**](projects/05-fpga-bnn-xnor-popcount) | A 32-element MAC replaced by XNOR + popcount; random-stream testbench | Verilog, Python | 🟢 verified in CI |
| 06 | [**TinyML gesture recognition**](projects/06-tinyml-gesture-esp32) | TCN on 6-axis IMU data, INT8 TFLite (263 KB) for an Arduino Nano ESP32 | TensorFlow, TFLite Micro | ✅ 66 % accuracy (5 classes) |
| 07 | [**Gauge reader**](projects/07-gauge-reader) | Reading gauges automatically with edge AI | — | 🚧 code being added |

**Status key:** ✅ measured results committed · 🟢 verified by automated tests/simulation · 🟡 implemented, hardware measurements pending

## 🧪 Experiments and 📚 notes
- [`experiments/`](experiments): smaller studies (Winograd, FFT convolution, Strassen, HEFT scheduling, ring all-reduce, FedAvg, hardware-aware NAS, compression, FPGA cost models, TVM). Two bugs were found and fixed during the latest review.
- [`notes/`](notes): deployment decision guides (TensorRT vs TVM vs FPGA, Jetson, DLA), FPGA platform spec, micro-NPU survey.

## Repository layout
```
projects/      self-contained projects, each with README, code, tests and reproduction steps
experiments/   supporting studies and prototypes
notes/         study notes and decision guides
tools/         small helpers (latency budget, MAC estimator)
.github/       CI: Python tests, golden-model tests, RTL simulation (Icarus Verilog)
```

## Continuous integration
Every push runs:
- golden-model unit tests for both FPGA projects (pytest)
- RTL simulation of all five testbenches with Icarus Verilog (bit-exact comparison + cycle counts)
- CPU checks of the experiments (Winograd, FFT conv, ring all-reduce)
- a syntax check of all Python code

## Skills
CUDA C++ · cuBLAS · TensorRT · ONNX · PyTorch · TensorFlow Lite Micro · SystemVerilog / Verilog · fixed-point arithmetic · Python (NumPy, pandas) · C/C++ · CMake · Git / GitHub Actions

---
**Author:** Lakshika Padmamali · Embedded Engineer · [GitHub](https://github.com/LakshikaMahipala)
