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
