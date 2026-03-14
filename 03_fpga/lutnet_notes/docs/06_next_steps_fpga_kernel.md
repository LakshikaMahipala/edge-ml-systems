
To move from notes → working FPGA proof:

1) Implement the BNN primitive kernel:
- bitpack activations/weights
- XNOR
- popcount
- accumulate
- threshold (sign)

2) Verify against Python reference:
- exact match for binary dot products

3) Measure:
- cycles per output vector
- throughput at assumed f_clk
- resource proxy: LUTs + adders

This is the minimal evidence needed before full LUTNet experiments.
