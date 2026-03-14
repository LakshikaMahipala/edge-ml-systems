TensorRT Overview

What TensorRT is
- NVIDIA inference optimizer and runtime.
- Takes a trained model (usually ONNX) and builds an optimized engine for a target GPU.

Core idea: build-time vs run-time
1) Build-time (offline):
   - Parse ONNX
   - Select kernel implementations (tactics)
   - Fuse layers where possible
   - Pick precision (FP32/FP16/INT8)
   - Serialize engine (.plan)

2) Run-time (online):
   - Load engine
   - Allocate buffers
   - Execute inference in a loop

Why we benchmark FP32 first
- Baseline correctness and pipeline sanity.
- Then we apply FP16/INT8 and compare:
  accuracy delta vs latency gain.

Latency pitfalls
- Warmup is mandatory (GPU clocks + caches + lazy allocations).
- Always separate:
  - H2D copy time
  - GPU compute time
  - D2H copy time
  - end-to-end time

What to record
- p50/p99 latency per inference
- throughput (images/sec) for batch sizes
- engine build settings (workspace, tactics, precision, batch)

Deliverable target
- A repeatable script pipeline:
  PyTorch -> ONNX -> TensorRT engine -> benchmark -> JSON results.
