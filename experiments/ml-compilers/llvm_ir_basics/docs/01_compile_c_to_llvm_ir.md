Compile C to LLVM IR (runbook)

Prereqs
- clang installed (LLVM toolchain)

Emit LLVM IR (human readable .ll)
- clang -S -emit-llvm -O0 src/vec_add.c -o outputs/vec_add_O0.ll
- clang -S -emit-llvm -O3 src/vec_add.c -o outputs/vec_add_O3.ll

Emit bitcode (.bc) then disassemble
- clang -c -emit-llvm -O3 src/vec_add.c -o outputs/vec_add.bc
- llvm-dis outputs/vec_add.bc -o outputs/vec_add_from_bc.ll

Optional: view assembly too
- clang -S -O3 src/vec_add.c -o outputs/vec_add_O3.s

Why O0 vs O3
- O0 keeps IR close to source (more loads/stores).
- O3 applies optimizations (vectorization, unrolling, mem2reg, etc.).
Comparing them teaches what compilers actually do.
