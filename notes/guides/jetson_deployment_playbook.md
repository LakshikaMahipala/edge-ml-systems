Jetson Deployment Playbook 

Goal
Turn our benchmarks into a deployable Jetson workflow.

Step 0 — Decide packaging model
Option A: Native JetPack install (SDK Manager)
- NVIDIA documents using SDK Manager to install JetPack on supported Jetson devices (including Orin Nano/AGX Orin).  [oai_citation:6‡NVIDIA Docs](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html?utm_source=chatgpt.com)

Option B: Containers (recommended for reproducibility)
- NVIDIA provides L4T JetPack containers that bundle JetPack’s accelerated libraries.  [oai_citation:7‡catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack?utm_source=chatgpt.com)
- There are also specific runtime containers like l4t-tensorrt for deployment.  [oai_citation:8‡catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt?utm_source=chatgpt.com)

Step 1 — Convert model
- PyTorch -> ONNX (from tensorrt_bench/scripts/export_onnx.py)

Step 2 — Choose runtime
- Best performance path on Jetson:
  ONNX -> TensorRT engine -> run inference
- TensorRT is NVIDIA’s inference SDK and supports mixed precision like FP16/INT8.  [oai_citation:9‡NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html?utm_source=chatgpt.com)

Step 3 — Consider DLA (if available)
- TensorRT supports targeting DLA for specific layers/workloads.  [oai_citation:10‡NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dla.html?utm_source=chatgpt.com)
- We will only use DLA if:
  - model operator set is compatible
  - throughput/latency improves under our constraints

Step 4 — Measure properly
- Always record:
  - warmup + p50/p99 latency
  - end-to-end time (include preprocessing and copies when relevant)
- Use our existing measurement discipline (inference_bench + tensorrt_bench)

Step 5 — Package
- Prefer containers for “works on my machine” elimination:
  - base: l4t-base
  - runtime: l4t-tensorrt or l4t-jetpack depending on needs  [oai_citation:11‡catalog.ngc.nvidia.com](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt?utm_source=chatgpt.com)
