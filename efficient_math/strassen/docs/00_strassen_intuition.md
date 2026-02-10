Strassen intuition (Week 11 Day 1)

Problem
Naive matrix multiply for NxN uses:
- multiplications: N^3
- additions: ~N^3

Strassen’s trick
Split matrices into 2x2 blocks and compute the product using:
- 7 multiplications instead of 8
- more additions/subtractions

Why this matters
Multiplications are more expensive than additions in many contexts.
Reducing multiplications can reduce runtime, energy, or hardware DSP usage.

Hidden truth (systems view)
Modern CPUs/GPUs are optimized for GEMM:
- cache blocking
- vectorization
- fused multiply-add
So Strassen often loses for small/medium sizes because:
- extra additions increase memory traffic
- recursion overhead hurts locality
- highly optimized BLAS beats naive code

So why learn it?
Because it teaches:
- algorithmic vs hardware tradeoffs
- when “fewer ops” does not mean faster
- how to reason about regimes and crossover points
