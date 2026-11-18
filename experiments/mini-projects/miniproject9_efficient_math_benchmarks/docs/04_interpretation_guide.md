# Interpretation guide (how to write conclusions)

When you run benchmarks later, do NOT just report times.
You must explain regimes:

1) Identify the crossover point
- For FFT: smallest K where FFT beats naive (as N grows)
- For Strassen: does it ever beat numpy.dot? if not, why?
- For SVD: rank where error acceptable and speed improves

2) Explain in systems terms
- memory traffic
- cache locality
- recursion overhead
- library optimization (BLAS/FFT backends)

3) Translate to FPGA terms
- which resource is limiting (DSP vs BRAM vs bandwidth)
- which pipeline stage dominates
- whether transforms are amortized
