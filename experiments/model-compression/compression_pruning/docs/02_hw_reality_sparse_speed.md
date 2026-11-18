# Hardware reality: why sparsity doesn't always speed up

Dense hardware (CPU SIMD, GPU Tensor Cores, FPGA MAC arrays) is built for:
- regular memory access
- dense compute pipelines

Sparse compute introduces:
- irregular indexing
- load imbalance
- branch/control overhead
- poor vector utilization

Therefore:
- unstructured pruning helps storage immediately
- but speed only improves if you have a sparse engine

FPGA:
- structured pruning reduces DSP lanes needed and simplifies datapath
- unstructured pruning needs sparse decoding logic (often not worth it early)
