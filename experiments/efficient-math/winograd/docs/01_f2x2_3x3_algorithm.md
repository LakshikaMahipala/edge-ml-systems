Winograd F(2x2, 3x3) algorithm sketch

We compute a 2x2 output tile from a 4x4 input tile and a 3x3 filter.

Steps
1) Filter transform (offline, once per filter):
   U = G * g * G^T

2) Input transform (per 4x4 patch):
   V = B^T * d * B

3) Elementwise multiply:
   M = U ⊙ V

4) Output transform:
   Y = A^T * M * A

Why 4x4 input?
Because to generate 2x2 outputs with a 3x3 kernel, you need a 4x4 receptive field.

Offline filter transform
U depends only on weights, so it can be precomputed and reused.
That is one of the big wins in inference.
