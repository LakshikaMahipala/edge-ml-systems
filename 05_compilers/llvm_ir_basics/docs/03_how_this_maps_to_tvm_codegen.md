How LLVM relates to TVM codegen

TVM pipeline (simplified)
Relay (graph IR) ->
TIR (tensor/kernel IR) ->
LLVM codegen (for CPU targets) ->
machine code

Mapping intuition
- Relay: conv/relu/dense nodes
- TIR: loops implementing conv/dense with tiling/vectorization
- LLVM IR: low-level representation of those loops after lowering

Why we care
When TVM targets CPU, it often uses LLVM to produce binaries.
So LLVM IR is the “last readable stop” before machine code.

Where FPGA fits
FPGA backends usually don't generate LLVM IR for datapaths.
But LLVM still matters for:
- host runtime
- command stream scheduling
- CPU reference backend
- toolchain glue

Professional takeaway
If you can read LLVM IR, you can debug why “codegen didn’t optimize”.
That’s a core skill in ML systems roles.
