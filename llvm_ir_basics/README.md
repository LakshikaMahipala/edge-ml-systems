LLVM IR Basics 

Goal
- Learn LLVM IR fundamentals (SSA, blocks, phi, loads/stores)
- Emit LLVM IR from simple C kernels at O0 and O3
- Understand how this connects to TVM lowering/codegen

Docs
- docs/00_what_is_llvm_ir.md
- docs/01_compile_c_to_llvm_ir.md
- docs/02_how_to_read_llvm_ir.md
- docs/03_how_this_maps_to_tvm_codegen.md

Run later
./scripts/emit_ll.sh
./scripts/emit_opt_ll.sh

Expected outputs
- outputs/*_O0.ll and outputs/*_O3.ll
- outputs/vec_add_opt.ll
