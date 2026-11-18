# TensorRT vs TVM vs FPGA — decision matrix (Week 10)

## One sentence summaries
- **TensorRT**: best “make my NVIDIA inference fast now” toolchain (fusion + precision + kernels) when you deploy on NVIDIA GPUs/DLA.
- **TVM/meta-schedule**: best “research + portability + learned scheduling” compiler when you need custom schedules or non-NVIDIA targets.
- **FPGA kernels**: best “custom datapath + deterministic streaming” option when you can amortize dev effort and your I/O path supports it.

---

## What each is optimizing

### TensorRT
Optimizes:
- kernel selection (cuDNN/cuBLAS/etc)
- layer fusion
- precision (FP16/INT8)
- memory planning
- engine building for a fixed GPU architecture

Great when:
- target is NVIDIA GPU/Jetson/DLA
- model is supported well (common CNNs, transformers with plugins)
- you want high performance with low engineering time

Limits:
- portability is low (tied to NVIDIA ecosystem)
- unsupported ops require plugins
- performance depends on batching and GPU occupancy

---

### TVM / Meta-Schedule
Optimizes:
- how loops are tiled, ordered, and lowered
- codegen to multiple backends (LLVM, CUDA, etc)
- autotuning policies (learned cost models)
- portability and research extensibility

Great when:
- you need fine control over schedules
- you want portability across devices
- you want to experiment with learned cost models / HW-aware compilation

Limits:
- setup complexity and tuning time
- real-world wins depend on good measurement + stable environment
- operator coverage vs production frameworks must be validated

---

### FPGA kernels
Optimizes:
- custom dataflow (streaming pipelines)
- fixed-point arithmetic and bespoke parallelism (unroll/II)
- deterministic latency and low jitter
- energy efficiency (often) if I/O is handled correctly

Great when:
- you can define a stable operator contract
- you can stream data efficiently (PCIe/Ethernet, not slow UART)
- you need deterministic latency, low tail, or tight power budgets

Limits:
- dev time + verification cost
- performance is often bottlenecked by I/O or memory bandwidth
- requires careful resource budgeting (DSP/BRAM) and tooling

---

## Practical positioning summary

If your deployment is:
- **NVIDIA GPU / Jetson** → start with **TensorRT**
- **non-NVIDIA / research scheduling** → use **TVM**
- **strict real-time / custom pipeline / power** → consider **FPGA kernels**, but only if I/O supports the throughput
