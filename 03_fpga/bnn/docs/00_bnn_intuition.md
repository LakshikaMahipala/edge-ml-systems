# BNN intuition 

Binary Neural Networks restrict values to 1-bit:
- weights W ∈ {+1, -1}
- activations a ∈ {+1, -1} (sometimes only weights are binarized)

Why this matters
- multiplications become cheap bit operations
- on FPGA, XNOR + popcount can replace MACs
- huge memory reduction (32-bit → 1-bit)

The trade-off
- representational capacity drops
- training becomes harder
- accuracy often lower unless architecture/training tricks are used

BNNs are a "hardware-first" technique.
