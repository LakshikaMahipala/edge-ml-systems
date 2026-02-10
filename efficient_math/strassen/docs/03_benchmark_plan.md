Benchmark plan 

Matrix sizes
N in {64, 128, 256, 512, 1024}

Variants
1) numpy.dot (highly optimized baseline)
2) naive triple-loop (slow, teaching reference)
3) Strassen (this implementation), with thresholds:
   N0 in {32, 64, 128}

Metrics
- runtime ms (p50/p99 over multiple runs)
- correctness error: max_abs_error vs numpy.dot
- memory: approximate temp allocation count (qualitative notes)

Expected outcome
- numpy.dot wins most sizes (baseline is optimized BLAS)
- Strassen may beat naive triple-loop at moderate N
- Strassen may or may not beat numpy.dot depending on machine/BLAS
