# Week 11 Summary: Which math trick helps which layer?

This guide translates efficient-math ideas into actionable decisions for inference acceleration.

---

## Decision table (high-level)

### 1) Dense / GEMM / Fully-connected (FC)
Best levers:
- **Low-rank SVD** (practical, common)
- (Sometimes) structured pruning / quantization (outside Week 11)

Consider Strassen only if:
- GEMM is extremely large
- and you control implementation (rare in practice)

Why low-rank helps:
- reduces MACs from IN*OUT → IN*r + r*OUT
- maps well to FPGA unroll factors (choose r aligned to hardware)

Failure modes:
- rank too high → no savings
- intermediate vector bandwidth becomes bottleneck

---

### 2) 3×3 convolution (standard CNN blocks)
Best lever:
- **Winograd** (classic win for 3×3)

Why it helps:
- reduces multiplications per output tile
- filter transform can be done offline

Regime where it shines:
- moderate/large channel counts
- many tiles (large feature maps)

Failure modes:
- fixed-point/INT8 scaling can amplify error (fractions in transforms)
- BRAM pressure for tile buffering on FPGA
- small channels or tiny feature maps → overhead dominates

---

### 3) Large-kernel convolution (e.g., 7×7, 11×11) or long 1D signals
Best lever:
- **FFT convolution**

Why it helps:
- direct cost grows with kernel size; FFT cost grows like N log N

Regime where it shines:
- large K (kernel length/area)
- large spatial size / long signal length

Failure modes:
- for small kernels (3×3), overhead dominates
- FFT on FPGA is heavy: buffering + control + scaling at each stage

---

### 4) “General matmul” on CPU/GPU with optimized BLAS
Baseline reality:
- **numpy.dot / cuBLAS / cuDNN are extremely optimized**
So algorithmic tricks (Strassen) often lose unless in a specialized regime.

Practical takeaway:
- prefer library-level optimizations (tiling, fusion, precision)
- only use math tricks if you can validate crossover on your target machine

---

## The regime rule (the hidden systems truth)

A math trick helps only if:
1) it reduces the dominant cost term (compute vs bandwidth vs I/O)
2) the overhead is amortized (many channels/tiles, reused weights)
3) numerical stability remains acceptable under your precision constraints

If I/O dominates (UART regime):
- none of these tricks will improve end-to-end latency meaningfully.
You must upgrade I/O before claiming acceleration.

---

## FPGA mapping (what we would implement first)

If building FPGA inference kernels:
1) **Low-rank SVD FC**: easiest to get measurable DSP savings with clean datapaths.
2) **Winograd 3×3**: high value, but fixed-point and buffering must be engineered carefully.
3) **FFT conv**: only for signal-processing style workloads; not the first CNN lever.

---

## Link to our estimator
Use:
- `fpga_cost_model_v1/` for BASE vs LOWRANK ranking and for budgeting IO vs compute.
Extend it later with:
- Winograd transform overhead terms per tile
- FFT transform overhead per signal block
