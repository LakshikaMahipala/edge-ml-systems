What TVM does (practical view)

TVM is an optimizing compiler framework for ML inference.

It provides:
1) Frontends
- import from ONNX, PyTorch (via tracing/export), TensorFlow, etc.

2) Graph optimizations
- rewrite the model graph for efficiency

3) Scheduling + autotuning
- search over implementation choices (tiling, unrolling, vectorization)
- measure on target hardware
- learn a cost model

4) Code generation
- generate C/CUDA/LLVM binaries for many targets
- can target exotic backends if you implement codegen hooks

Core loop (autotuning idea)
IR → propose schedule → generate code → run/measure → update cost model → repeat
