What is Relay?

Relay is TVM’s high-level graph IR for neural networks.
It represents models as a functional program:
- explicit operators
- explicit dataflow
- types/shapes (often inferred)

What Relay enables
- graph-level optimizations:
  - constant folding
  - dead code elimination
  - operator fusion
  - layout transforms
  - quantization-aware graph rewrites (in pipelines)

Where Relay sits in the pipeline
Model (PyTorch/ONNX) → Relay IR → optimized Relay → lowered to kernel IR (TIR) → codegen → binary

Why we care for FPGA work
A compiler needs:
- a stable op set (our platform spec)
- a cost model (our FPGA timing model)
Relay is where op selection and fusion decisions can be influenced.
