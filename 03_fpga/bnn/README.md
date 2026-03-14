Binary Neural Networks 

Key idea
- Replace multiply-accumulate with XNOR + popcount by binarizing weights/activations.
- Training requires STE (straight-through estimator).
- Accuracy often drops; we use BN and keep last layer FP32 to reduce collapse.

Next days
- LUTNet mapping concepts
- FPGA XNOR-popcount kernel
