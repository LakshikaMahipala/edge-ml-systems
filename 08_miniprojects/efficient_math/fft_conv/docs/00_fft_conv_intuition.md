FFT convolution intuition 

Convolution theorem
conv(x, h) in time/spatial domain equals:
IFFT( FFT(x) * FFT(h) )

Why it can be faster
Direct convolution cost:
- 1D: O(N*K)
- 2D: O(H*W*KH*KW)

FFT convolution cost:
- FFT of x: O(N log N)
- FFT of h: O(N log N) (offline if h is fixed)
- multiply: O(N)
- IFFT: O(N log N)

It wins when K is large (or signal is huge).
It loses when kernel is small (3x3) because transform overhead dominates.

Offline filter transform
Like Winograd, FFT(h) can be precomputed and reused in inference.
