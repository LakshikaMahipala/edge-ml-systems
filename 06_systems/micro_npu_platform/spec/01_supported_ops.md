Supported operators (v0)

We define a strict operator set so compilation and runtime are predictable.

Core ops (v0)
1) INT8_FC
- y = requant(Wx + b)
- Inputs: int8 activations
- Weights: int8
- Accum: int32
- Output: int8

2) INT8_DWCONV1D_K3
- depthwise 1D conv, K=3, stride=1
- Per-channel kernel
- Accum int32, output int8

3) Activation: RELU_INT8
- clamp negative to 0 (assuming symmetric quant or zero-point handling simplified)

4) (Optional next) INT8_POINTWISE_1x1
- channel mixing after depthwise
- essentially FC per spatial position

Constraints
- static shapes only
- batch=1 only (v0)
- all tensors are contiguous arrays in row-major order

Rationale
This matches MobileNet-style blocks:
DWConv -> ReLU -> PWConv -> ReLU
