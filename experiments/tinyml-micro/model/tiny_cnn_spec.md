Tiny CNN spec (for later training/export)

Input: 1x28x28 grayscale (or 1x32x32 if you prefer)
Backbone:
- Conv2D 3x3, C=8, stride=1 + ReLU
- MaxPool 2x2
- Conv2D 3x3, C=16, stride=1 + ReLU
- MaxPool 2x2
Head:
- Flatten
- FC 32 + ReLU
- FC num_classes

Quantization:
- INT8 weights + INT8 activations
- Per-tensor quantization for simplicity

Why this model
- operator set is MCU friendly (conv/pool/FC)
- small enough for TFLite Micro
- maps cleanly to CMSIS-NN kernels
