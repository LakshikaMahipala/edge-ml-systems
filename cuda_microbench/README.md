CUDA Microbench — Vector Add

Goal
- Validate CUDA toolchain and the core execution flow:
  malloc -> H2D -> kernel -> sync -> D2H -> verify

Build 
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

Run 
./vector_add 1048576

Record (docs/metrics.md)
- correctness: OK
- timing: add later (we will add CUDA events timing on Day 2+)

GEMM baselines (correctness first)

Build 
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

Run 
./gemm_naive 512 512 512
./gemm_cublas 512 512 512

Expected
- Both print max_abs_err and max_rel_err vs CPU reference.
- Timing will be added next (CUDA events) for fair comparison and GFLOP/s.

Tiled GEMM (shared memory)

Run 
./gemm_tiled 512 512 512

Record
- correctness vs CPU
- (timing added later)
