# XNOR + popcount view

If we encode:
+1 → 1
-1 → 0

Then multiplication between binary values becomes XNOR:
(+1)*(+1)=+1, (-1)*(-1)=+1, (+1)*(-1)=-1

Dot product can be computed by:
dot(a, w) ≈ (2*popcount(xnor(a_bits, w_bits)) - n)

So:
- XNOR gives matches
- popcount counts matches
- map back to signed sum

This is the FPGA-friendly core.
