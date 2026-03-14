DWConv1D INT8 v0 

Goal
Implement depthwise 1D convolution (INT8) for Micro-NPU style blocks.

Spec
- C=4, L=16, K=3
- per-channel depthwise convolution
- int32 accumulation + SHIFT requant + saturate to int8

Run later
./scripts/sim_iverilog.sh
Expected: PASS tb_dwconv1d_int8

Next
- Upgrade to streaming shift-register implementation (data reuse)
- Extend to 2D depthwise conv + pointwise conv
