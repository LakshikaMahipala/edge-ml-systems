IR basics (Intermediate Representation)

What an IR is
An IR is a structured representation of a program that is easier to analyze and optimize than source code.

Why compilers use IR
- source code is messy (syntax, high-level abstractions)
- machine code is too low-level
IR sits in the middle: enough structure to optimize, enough detail to generate code.

Two key properties
1) Explicit dataflow
- which ops depend on which results
2) Explicit types/shapes (in ML compilers)
- tensor shapes and dtypes drive memory planning and kernel selection

Why ML needs compilers
Neural nets are huge compute graphs.
Performance comes from:
- operator fusion
- layout transforms
- tiling and vectorization
- memory reuse
- choosing the right backend kernel

All those require a graph-like IR.
