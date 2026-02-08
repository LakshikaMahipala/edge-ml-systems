Operator set + Memory model (how to think like a hardware ML scientist)

Operator set
The single biggest practical constraint in deployment is not FLOPs — it is whether your device/compiler supports your ops.

Commonly supported (almost everywhere)
- Conv2D / DepthwiseConv2D
- GEMM / Fully connected
- ReLU / ReLU6
- Pooling
- BatchNorm folding
- Reshape / Transpose (limited forms)

Often painful / limited
- dynamic shapes
- advanced attention ops
- custom normalization layers
- irregular control flow

Memory hierarchy (why “small” accelerators win)
Compute is cheap; memory movement is expensive.

Typical hierarchy
- Registers (fastest)
- Local SRAM/BRAM (fast, small)
- Shared SRAM / cache
- DRAM (slow, power heavy)

Hardware implication
A good accelerator design increases:
- data reuse
- locality
- streaming overlap
and reduces:
- round-trips to DRAM
- format conversions
- host-device copies

The “roofline” intuition (no math required today)
Performance is limited by either:
- compute throughput (MACs/s)
OR
- memory bandwidth (bytes/s)

Many real deployments are memory-bound.
That’s why we treat I/O first-class (see performance_notebook).
