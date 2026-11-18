Nsight Systems vs Nsight 

What each tool answers

1) Nsight Systems (nsys) = “Timeline / System view”
Use it when you want to know:
- Where time goes end-to-end (CPU vs GPU vs memcpy)
- Whether kernels overlap with memcpy (pipelines)
- Whether you accidentally synchronize too often
- CPU thread scheduling delays
Outputs:
- a timeline of CPU threads, GPU kernels, memcpy, NVTX ranges

2) Nsight Compute (ncu) = “Kernel microscope”
Use it when you want to know:
- Is the kernel compute-bound or memory-bound?
- Achieved occupancy
- Memory throughput, cache hit rates
- Warp execution efficiency
- Instruction mix
Outputs:
- per-kernel performance metrics and roofline-style hints

Golden rule: profile in two phases
Phase A: nsys first (macro)
- find the expensive kernels and sync points
Phase B: ncu second (micro)
- zoom into the top 1–3 kernels and optimize them

What to record in this project (minimum)
A) From nsys (timeline)
- Total runtime breakdown:
  H2D copies time
  Kernel time
  D2H copies time
  CPU-side overhead / sync points
- List top kernels by time (name + %)
- Count of cudaDeviceSynchronize calls (too many = bad)

B) From ncu (kernel metrics)
For gemm_naive, gemm_tiled, conv_lite_gemm:
- Achieved occupancy (%)
- SM throughput (or “compute utilization”)
- DRAM throughput (GB/s)
- L2 hit rate (if available)
- Warp execution efficiency
- Whether memory is coalesced (load/store efficiency)

How to interpret quickly
- If occupancy is low: likely too many registers or shared memory per block
- If DRAM throughput is near peak but SM throughput is low: memory-bound
- If SM throughput is high but DRAM is low: compute-bound
- If warp execution efficiency is low: divergence or poor mapping
- If a lot of memcpy time: data movement dominates; Amdahl applies

Run commands (later, examples)

Nsight Systems (timeline):
nsys profile -o nsys_gemm_tiled ./gemm_tiled 1024 1024 1024

Nsight Compute (kernel metrics):
ncu --set full --target-processes all ./gemm_tiled 1024 1024 1024

Notes
- Always run Release builds.
- Always do warmup iterations before timing.
- For fair timing, do not include H2D/D2H unless you explicitly want end-to-end.
