SVD factorization for compression

SVD:
W = U Σ V^T

Rank-r approximation (keep top r singular values):
W_r = U_r Σ_r V_r^T

Implementation trick
Define:
A = V_r Σ_r   (in x r)
B = U_r^T     (r x out)   [careful with shapes depending on convention]

Then:
x W_r = (x A) B

We prefer to fold Σ into one side to keep one matrix scaled and the other orthonormal.
