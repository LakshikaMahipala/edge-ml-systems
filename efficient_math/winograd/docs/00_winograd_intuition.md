Winograd intuition 

Target
Speed up small-kernel convolution, especially 3x3.

What conv2d costs
For each output pixel, 3x3 conv uses 9 multiplications per input channel.

Winograd idea
Compute a small output tile (e.g., 2x2 outputs) using transforms:
- transform input tile
- transform filter
- elementwise multiply in transform domain
- inverse transform to output

Benefit
For F(2x2,3x3):
- multiplications per output tile drop compared to naive conv
- more additions, but additions are cheaper on many platforms

Big hidden point
Winograd is beneficial only if:
- transform overhead is amortized over many channels / many tiles
- you have good memory layout (NCHW/NHWC choices matter)
- numerical stability is acceptable (FP16 and INT8 can be tricky)
