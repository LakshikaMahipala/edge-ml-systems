CMSIS-NN: what it is and why it matters

What CMSIS-NN is
CMSIS-NN is a set of highly-optimized neural-network kernels for Arm Cortex-M CPUs.
It provides fast INT8/INT16 implementations of common ops (conv, depthwise conv, FC, pooling, activation).

Why it exists
On microcontrollers:
- no GPU
- low clock
- tight RAM/Flash
- integer arithmetic is much cheaper than float

So performance depends on:
- quantization (INT8)
- good kernels (CMSIS-NN)
- careful memory layout

How it relates to TFLite Micro
TFLite Micro (TFLM) is a micro inference runtime.
When built with CMSIS-NN enabled, many operators are dispatched to CMSIS-NN kernels.

In our repo
We will:
- build a TFLM-style C++ inference skeleton
- keep model INT8
- estimate cost (MACs) today
Later we will run on a real MCU/board.
