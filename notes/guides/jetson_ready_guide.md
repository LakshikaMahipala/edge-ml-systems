Jetson-ready Guide 

Purpose
This guide describes the exact deployment path we will use when a Jetson device is available.
It is written so that the repo can be executed without guessing.

Baseline assumptions
- We will use a JetPack 6.x baseline on Orin-family devices (common deployment baseline).
- TensorRT is the primary inference runtime for GPU acceleration.
- Containers are preferred for reproducibility.

Step-by-step plan

Step 0 — Identify Jetson + JetPack
- Record:
  - Jetson module (Orin Nano / Orin NX / AGX Orin / etc.)
  - JetPack version
  - TensorRT version
  - CUDA version
Reference: JetPack release notes list versions for CUDA / TensorRT / cuDNN.  [NVIDIA JetPack release notes]

Step 1 — Decide packaging method
Option A: Native (SDK Manager)
- Use NVIDIA SDK Manager to flash/install JetPack.  [JetPack install docs]

Option B: Container (recommended)
- Use NVIDIA L4T containers (l4t-jetpack / l4t-tensorrt).  [NGC container catalog]

Step 2 — Bring our models to Jetson
- Use tensorrt_bench:
  - export ONNX
  - build FP32/FP16 engines
  - build INT8 engine with calibration cache
- Record engine build logs and parsed JSON summaries.

Step 3 — Benchmark properly
- Use warmup + iters and report p50/p99.
- Track both:
  - engine latency (trtexec)
  - end-to-end pipeline latency (inference_bench + video_demo)

Step 4 — Optional: DLA evaluation
- Try:
  - DLA + GPU fallback
  - DLA-only (compatibility test)
- Record which layers run on DLA vs GPU.

Step 5 — Results publishing
- Put final tables in:
  - docs/mini_project_3_trt_vs_ort_vs_pytorch.md
  - docs/week6_summary_jetson_fpga_video.md
- Commit JSON summaries (small) but not large binary engines.

References
- JetPack install/setup docs: https://docs.nvidia.com/jetson/jetpack/install-setup/
- JetPack release notes: https://docs.nvidia.com/jetson/jetpack/release-notes/
- L4T containers: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack
- L4T TensorRT runtime: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt
