CUDA Microbench Foundations

What was built
- vector_add: CUDA sanity kernel
- gemm_naive: baseline GEMM (correctness + CUDA event timing)
- gemm_tiled: shared-memory tiled GEMM (correctness + timing)
- gemm_cublas: vendor baseline (correctness + timing)
- conv_lite_gemm: conv lowering via im2col + GEMM (timing GEMM-only)

How to reproduce 
- Build: cuda_microbench/README.md
- Suite run: cuda_microbench/scripts/run_all.sh
- Parse + plot: scripts/parse_to_csv.py and scripts/plot_results.py

What to fill when local runs are possible
- docs/metrics.md:
  - avg_ms, GFLOP/s
  - speedups: tiled vs naive, cublas vs tiled
- docs/mini_project_2_report.md:
  - key takeaways + Nsight screenshots + reasons

Next 
- Begin TensorRT track while FPGA work starts (UART + fixed-point primitives).
