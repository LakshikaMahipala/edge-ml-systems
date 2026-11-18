# CUDA SGEMM Optimization: from Naive to Register Blocking

> Step-by-step FP32 matrix multiplication kernels in CUDA, each verified against cuBLAS and benchmarked across sizes, to show *why* each optimization makes GEMM faster.

**Status:** 🟡 Code complete and kernel indexing verified in emulation · GPU results pending (run the [Colab notebook](notebooks/run_on_colab.ipynb) on a free T4)

## Why this matters
Matrix multiplication (GEMM) is most of the compute in neural networks: fully-connected layers, attention and convolutions (via im2col) all reduce to it. Writing GEMM kernels by hand is the standard way to learn the GPU memory hierarchy: global memory → shared memory → registers.

## Kernels

| # | Kernel | Key idea | What limits it |
|---|---|---|---|
| 1 | `sgemm_naive` | One thread per output element; reads A and B from global memory | Global-memory bandwidth: ~1 load per FLOP |
| 2 | `sgemm_smem_tiled<32>` | Block loads 32×32 tiles of A and B into shared memory; each value reused 32 times | Shared-memory bandwidth and `__syncthreads` |
| 3 | `sgemm_register_blocked<64,64,8,4,4>` | Each thread computes a 4×4 output micro-tile held in registers: 8 shared-memory loads feed 16 FMAs | Approaches the FP32 compute limit |
| — | cuBLAS `cublasSgemm` | Vendor-tuned reference | — |

All kernels handle arbitrary M, N, K (boundaries are zero-padded), and every result is checked against cuBLAS before its time is reported.

### Arithmetic intensity, in numbers
For one output element with K = 1024:
- **Naive:** 2 × 1024 global loads (8 KB) for 2 × 1024 FLOPs → **0.25 FLOP/byte**. At the T4's ~320 GB/s that caps out near 80 GFLOP/s, about 1% of its ~8 TFLOP/s FP32 peak.
- **32×32 tiling:** each loaded value is used 32 times → **~8 FLOP/byte** from DRAM. Now shared memory becomes the bottleneck.
- **4×4 register blocking:** per inner step, 4 + 4 = 8 shared loads produce 4 × 4 = 16 FMAs (32 FLOPs) → 4 FLOP per shared load instead of 1.

## Results (NVIDIA T4)
_Pending: run `notebooks/run_on_colab.ipynb` and commit `results/gemm_t4.csv` and `results/gemm_t4.png`._

| Size | naive | smem_tiled_32 | register_blocked | cuBLAS |
|---|---|---|---|---|
| 1024 | – | – | – | – |
| 4096 | – | – | – | – |

## Repository layout
```
├── src/
│   ├── common.cuh         # error checks, CUDA-event timer, error metric
│   ├── gemm_kernels.cuh   # the three SGEMM kernels
│   ├── gemm_bench.cu      # size sweep, cuBLAS check, CSV output
│   └── vector_add.cu      # DRAM bandwidth baseline (roofline ceiling)
├── scripts/plot_results.py
├── notebooks/run_on_colab.ipynb
├── docs/notes_nsight.md   # how to profile with Nsight Systems / Compute
└── results/               # committed CSVs and plots
```

## How to reproduce
Requires the CUDA Toolkit (11.x or 12.x) and CMake ≥ 3.18.
```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75   # 75 = T4, 86 = RTX 30xx, 89 = RTX 40xx
cmake --build build -j
./build/vector_add                                  # bandwidth baseline
./build/gemm_bench 256,512,1024,2048,4096 5 20 | tee results/gemm.csv
python scripts/plot_results.py results/gemm.csv results/gemm.png
```
No local GPU? Open `notebooks/run_on_colab.ipynb` in Google Colab with a T4 runtime.

Output columns: `kernel, M, N, K, time_ms, gflops, pct_of_cublas, max_norm_err, correct`.
`max_norm_err` = max |C − C_cuBLAS| / max |C_cuBLAS|; a kernel passes when it is below 1e-4.

## Method notes
- Timing uses CUDA events around 20 launches after 5 warm-up launches; host↔device copies are excluded.
- Inputs are uniform random in [−1, 1] with fixed seeds, so runs are reproducible.
- The kernels' index math was checked against NumPy with a Python emulation on odd sizes (e.g. 130×129×17).

## Limitations and next steps
- [ ] Commit T4 results and Nsight Compute "Speed of Light" summaries
- [ ] Vectorized `float4` loads and a transposed A tile (avoids shared-memory bank conflicts)
- [ ] Double buffering (overlap the next tile's loads with compute)
- [ ] Tensor-core versions: FP16 WMMA and INT8 (T4 supports both)
