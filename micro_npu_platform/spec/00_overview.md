Micro-NPU Platform Spec 

Purpose
Define a minimal deployable “Micro-NPU platform” consisting of:
- supported operator set
- tensor + quantization conventions
- memory map and buffers
- command stream interface
- scheduler model (host/runtime)
- timing model hooks

Non-goals (v0)
- full compiler implementation
- dynamic shapes
- training
We focus on a stable inference platform.

Philosophy
- Keep Unity/UI thin; keep logic in services.
- Treat I/O + correctness + reproducibility as first-class.
- Prefer small, verifiable kernels that compose into graphs.

Targets (conceptual)
- MCU-class runtime (TFLite Micro-like)
- FPGA kernel backends (UART now, faster I/O later)
- CPU reference backend
