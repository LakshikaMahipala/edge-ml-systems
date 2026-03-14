Mini-project 2 — CUDA Microbench Suite 

What was built
- vector_add baseline
- gemm_naive baseline
- gemm_tiled shared-memory GEMM
- gemm_cublas reference (vendor-optimized)
- conv_lite_gemm (im2col + GEMM)

How to run (local)
1) Build
cd cuda_microbench
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

2) Run suite
cd ..
chmod +x scripts/run_all.sh
./scripts/run_all.sh

3) Parse and plot
python scripts/parse_to_csv.py results/run_*.txt results/summary.csv
python scripts/plot_results.py results/summary.csv results/plot_gflops.png

Results table (fill after run)
bench | avg_ms | GFLOP/s | max_abs_err | max_rel_err
- vector_add | | | |
- gemm_naive | | | |
- gemm_tiled | | | |
- gemm_cublas | | | |
- conv_lite_gemm | | | |

Key takeaways (fill after run)
- Speedup tiled vs naive:
- Gap between tiled and cuBLAS:
- Where time goes (nsys timeline):
- Kernel bottleneck reason (ncu metrics: occupancy vs memory throughput):

Next steps
- Improve tiled GEMM (register blocking, better tile size, avoid bank conflicts)
- Move im2col to GPU (avoid CPU overhead, avoid extra copies)
