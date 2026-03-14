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

Conv mapped to GEMM (conv-lite)

Run (later)
./conv_lite_gemm 3 32 32 8 3 3

Meaning
- C=3, H=W=32 input
- F=8 filters, 3x3 kernel
- valid conv, stride=1, pad=0
- implemented as im2col + cuBLAS GEMM

Record
- correctness max_abs_err/max_rel_err
- timing will be captured on Nsight day / microbench suite day

Profiling readiness 

Timing (CUDA events)
- Programs accept optional warmup and iters:
  ./gemm_tiled 1024 1024 1024 10 50

Nsight Systems (timeline)
nsys profile -o nsys_gemm_tiled ./gemm_tiled 1024 1024 1024 10 50

Nsight Compute (kernel metrics)
ncu --set full --target-processes all ./gemm_tiled 1024 1024 1024 10 50

Mini-project 2 — Suite mode

Run later:
cd cuda_microbench
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
cd ..
chmod +x scripts/run_all.sh
./scripts/run_all.sh

Parse + plot:
python scripts/parse_to_csv.py results/run_*.txt results/summary.csv
python scripts/plot_results.py results/summary.csv results/plot_gflops.png

Where to write conclusions:
docs/mini_project_2_report.md
