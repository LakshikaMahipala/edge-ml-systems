1D FFT convolution

We implement "full" convolution by:
- choosing padded length P >= N + K - 1
- FFT(x, P) and FFT(h, P)
- multiply and inverse FFT
- take first N+K-1 samples

This is the cleanest baseline to study crossover vs naive O(N*K).
