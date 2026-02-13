# Decision Guide: Quantization, Pruning, BNN, LUTNet (FPGA-aware)

This guide is written for making real engineering choices under constraints.

---

## 1) Quantization: PTQ vs QAT

### Use PTQ when:
- CNN-style model is stable (ResNet/MobileNet-like)
- small accuracy drop is acceptable
- you need fast deployment

### Use QAT when:
- PTQ accuracy drop is unacceptable
- model has activation outliers / sensitive layers
- you require strict INT8 accuracy

Hardware note:
- INT8 only helps if your backend has fast INT8 kernels (TensorRT / ORT EP / FPGA datapath)
- always specify IO regime (UART vs PCIe/Ethernet) before claiming speedup

Repo evidence:
- QAT toy project exists (run later for metrics)

---

## 2) Pruning: unstructured vs structured

### Unstructured pruning:
- great for compression (storage)
- speedup only with sparse kernels (rare, needs very high sparsity)

### Structured pruning:
- changes tensor shapes (neurons/channels removed)
- enables speedup on normal dense hardware (CPU/GPU/FPGA)

Use structured pruning when your goal is speed/area, not just model size.

Repo evidence:
- unstructured + structured pruning scripts exist (run later)

---

## 3) BNN: when to use it

BNN is a hardware-first technique:
- compute becomes XNOR + popcount + threshold
- model footprint drops dramatically
- FPGA LUT fabric can execute it efficiently

BNN is worth considering when:
- you are bandwidth/power constrained
- you target FPGA or micro-accelerator style hardware
- your accuracy target tolerates some drop or you can widen the model

Repo evidence:
- BNN Python model exists
- FPGA XNOR-popcount dot RTL exists with Python golden model

---

## 4) LUTNet: where it fits

LUTNet direction:
- replace parts of BNN compute graphs with LUT truth tables directly
- limited by LUT input size k (high fan-in problem)
- needs careful partitioning and training tricks

This is a “next level” after proving BNN kernel correctness and measuring throughput.

Repo evidence:
- mapping notes exist, ready for later experimentation

---

## 5) Evidence rules (credibility)
- no speedup claim without measurement context
- unstructured sparsity ≠ speedup by default
- FPGA claims require bit-exact validation vs golden model
