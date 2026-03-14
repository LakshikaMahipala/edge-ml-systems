How to read LLVM IR (cheat sheet)

Functions
define <ret_type> @func_name(args...) { ... }

SSA values
%0, %1, %tmp are SSA registers.
Each SSA name is assigned once.

Basic blocks
entry:
  ...
  br label %loop

Memory vs registers
- alloca: stack allocation
- store: write memory
- load: read memory
Optimizations often remove loads/stores by promoting allocas to SSA registers (mem2reg).

Loops
Typically appear as:
- phi nodes (merge values from multiple predecessors)
- conditional branches (br i1 ...)

Vectors and SIMD
Look for:
- <4 x float> or <16 x i8> vector types
- llvm.* intrinsics
- shufflevector

What you should learn from IR
- Did the compiler vectorize?
- Did it unroll loops?
- Did it remove redundant loads/stores?
- Where are the bottlenecks likely (memory vs compute)?
