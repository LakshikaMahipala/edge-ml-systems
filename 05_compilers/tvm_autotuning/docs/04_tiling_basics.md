Tiling basics 

What tiling is
Tiling splits a large loop nest into smaller blocks (tiles).
Example for GEMM:
C[i,j] += A[i,k] * B[k,j]
We tile i,j,k into blocks so that a small submatrix fits in cache or shared memory.

Why tiling matters
Modern hardware is usually bandwidth-limited, not compute-limited.
Without tiling:
- you reload the same data from DRAM many times
With tiling:
- you reuse data from faster memory (cache/shared) many times

Intuition (roofline)
Performance is limited by:
min(peak_compute, bandwidth * arithmetic_intensity)
Tiling increases arithmetic intensity by increasing reuse.

In TVM scheduling
Tiling is expressed via:
- split loops into outer/inner
- reorder loops
- cache_read/cache_write into faster scopes
- vectorize inner loops
- unroll small loops

What “good tiles” look like
- inner tile fits in L1/L2 cache (CPU) or shared memory (GPU)
- enough parallel work per tile
- avoids too much overhead (too-small tiles are bad)
