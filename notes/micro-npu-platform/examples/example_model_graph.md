Example model graph (MobileNet-like block, simplified)

Input X [C=4, L=16] int8
1) DWCONV1D_K3 -> Y1 [4, 14]
2) RELU -> Y2 [4, 14]
3) FC (pointwise-like) per position (future)
For v0 we stop after DWCONV + RELU to match our implemented kernels.
