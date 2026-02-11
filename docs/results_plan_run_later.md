# results plan (run later)

When local execution is possible:

1) Run each benchmark script:
- Strassen: efficient_math/strassen/src/benchmark_strassen.py
- Winograd: efficient_math/winograd/src/benchmark_winograd.py
- FFT conv: efficient_math/fft_conv/src/benchmark_fft_conv.py
- Low-rank: efficient_math/low_rank_svd/src/benchmark_low_rank.py

2) Consolidate:
- Mini-project 9 unify schema:
  miniproject9_efficient_math_benchmarks/src/unify_schema.py

3) Write conclusions:
- Identify crossovers (N/K thresholds)
- Explain in systems terms (cache, memory, overhead)
- Translate to FPGA terms (DSP/BRAM/bandwidth/II)

4) Update:
- fpga_cost_model_v1 assumptions if measurements contradict proxies
