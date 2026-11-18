Depthwise vs Pointwise (why MobileNet is fast)

Standard convolution mixes channels:
- Each output channel is a sum over all input channels.
Cost ~ H*W*C_in*C_out*K*K

Depthwise convolution does NOT mix channels:
- Each input channel has its own small kernel.
Cost ~ H*W*C_in*K*K

Pointwise (1x1) convolution is then used to mix channels:
Cost ~ H*W*C_in*C_out

Depthwise-separable conv = depthwise + pointwise
It reduces compute dramatically compared to standard conv.

Why we start with depthwise 1D
It captures the same hardware patterns:
- sliding window
- data reuse via shift registers
- per-channel independent pipelines
