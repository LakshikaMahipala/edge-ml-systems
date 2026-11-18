Complexity + regimes

Naive
T(N) = 8*T(N/2) + O(N^2)  => O(N^3)

Strassen
T(N) = 7*T(N/2) + O(N^2)  => O(N^(log2 7)) ≈ O(N^2.807)

Regimes
- For very large N, Strassen can win asymptotically.
- For moderate N, constants dominate and it may lose.
- Practical implementations use a crossover threshold:
  if N <= N0: use classical blocked GEMM

In hardware terms
- Strassen reduces multiply count, potentially saving DSPs.
- But increases adds and requires more temporary storage.
- For FPGA, memory/BRAM pressure can dominate.
