Crossover and regimes

We measure runtime for:
- Naive 1D conv O(N*K)
- FFT 1D conv O(P log P)

Sweep:
N in {256, 512, 1024, 2048, 4096, 8192}
K in {3, 7, 15, 31, 63, 127, 255}

Expected:
- For small K (3,7): naive faster
- For large K (>=63 or >=127): FFT wins for large N

Important factors:
- FFT implementation quality (NumPy uses optimized FFT backends)
- real-valued FFT (rfft) reduces cost
- padding to power-of-two can matter
