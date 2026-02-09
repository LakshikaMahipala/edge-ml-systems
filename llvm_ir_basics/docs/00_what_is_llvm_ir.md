LLVM IR: what it is (and why ML people care)

LLVM IR is a low-level intermediate representation used by the LLVM compiler toolchain.
It sits between:
- high-level code (C/C++/Rust/Swift...)
and
- machine code (x86/ARM/RISC-V...)

Key properties
- SSA form (Static Single Assignment): each value is assigned once.
- explicit control flow: basic blocks + branches.
- explicit memory ops: loads/stores (unless promoted to registers by optimizations).
- types are explicit: i32, float, pointers, vectors, etc.

Why this matters for ML compilers
Graph IR (Relay) decides what ops run.
Kernel IR (like TVM TIR) decides how ops run (loops/tiles).
Eventually, the compiler must emit real code:
- for CPU: LLVM IR -> machine code
- for accelerators: LLVM sometimes used for host stubs or runtime pieces

So LLVM IR is where “codegen becomes real”.
