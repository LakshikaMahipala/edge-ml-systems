# Unstructured vs structured pruning

## Unstructured
- remove individual weights by magnitude threshold
- model tensor shapes stay the same
- creates sparse matrices

Pros: easy, flexible, good compression
Cons: speedup not guaranteed

## Structured
- remove entire neurons (MLP), channels (CNN), heads (Transformer)
- tensor shapes shrink (dense kernels still work)

Pros: speedup on normal dense hardware (CPU/GPU/FPGA)
Cons: must handle shape changes + sometimes more accuracy drop per % removed
