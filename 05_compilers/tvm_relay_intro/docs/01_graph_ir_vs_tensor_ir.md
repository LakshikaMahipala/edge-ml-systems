Graph IR vs Tensor IR (critical distinction)

Graph IR (model-level)
- nodes = operators (conv, relu, matmul)
- edges = tensors flowing between ops
Used for:
- operator fusion
- constant folding
- memory planning
- scheduling across devices

Tensor IR (kernel-level)
- describes how one operator is computed (loops, tiles, vectorization)
Used for:
- generating fast kernels
- mapping to SIMD/GPU/FPGA
- autotuning schedules

Why this matters
Relay is primarily Graph IR (high-level).
TVM also has lower-level IRs for kernels (TensorIR / TIR).

Mental model
Graph IR decides "WHAT ops run and in what order"
Tensor IR decides "HOW each op is implemented efficiently"
