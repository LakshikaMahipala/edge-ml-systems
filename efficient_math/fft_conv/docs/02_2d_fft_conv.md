2D FFT convolution (single-channel)

We implement valid conv by:
- pad input and kernel to a common size (H+KH-1, W+KW-1) for full conv
- FFT2 both, multiply, IFFT2
- extract valid region

Reality
In deep learning libraries, FFT conv is used mainly in special regimes
(large kernels, large feature maps, or certain backprop computations).
