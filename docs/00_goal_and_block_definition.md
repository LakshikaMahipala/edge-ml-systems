Mini-project 6: MobileNet-like block 

Goal
Build and compare a minimal “MobileNet-like” block using our current pieces:
- Depthwise conv (DWConv1D K=3, INT8)
- ReLU
- (Optional) Pointwise mixing (not implemented in FPGA yet; we include as spec + MAC estimate)

Block v0 (what we actually implement)
Input X: int8[C=4][L=16]

Step 1: DWConv1D_K3 -> Y1: int8[4][14]
Step 2: ReLU_INT8 -> Y2: int8[4][14]

Block v1 (future extension)
Step 3: Pointwise 1x1 (mix channels) -> Y3: int8[C_out][14]

Why 1D?
It is the cleanest way to teach and verify:
- sliding window
- depthwise vs pointwise separation
- fixed-point behavior
- platform evaluation with I/O included

Outputs
- reference Python outputs (bit-approx)
- MAC estimates for micro runtime
- FPGA timing model estimates (UART + cycles)
- comparison narrative: where the bottleneck is and why
