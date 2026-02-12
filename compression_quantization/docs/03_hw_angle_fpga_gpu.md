# Hardware angle: GPU vs FPGA

GPU/Jetson:
- INT8 speedups depend on Tensor Cores / backend kernels
- TensorRT can fuse + quantize aggressively

FPGA:
- INT8 reduces DSP usage and memory bandwidth
- but scaling (s, z) and saturation must be implemented correctly
- accuracy failures often come from bad range selection and overflow
