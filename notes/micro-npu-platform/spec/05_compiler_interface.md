Compiler interface (v0)

Goal
Define how a trained model graph becomes a command stream + weights blob.

Inputs to compiler
- model graph (operators + tensor shapes)
- quantization parameters
- target constraints (operator set, buffer sizes)

Outputs
1) weights.bin (or .hex)
- packed int8 weights and int32 biases in a defined order
2) commands.json
- list of operator invocations (addresses + params)
3) metadata.json
- tensor sizes, required scratch, versioning

Compilation constraints (v0)
- only supported ops
- static shapes
- batch=1
- one linear execution order (no branching)

Validation step
- run CPU reference backend with same quant rules
- compare outputs bit-exactly for sample inputs
