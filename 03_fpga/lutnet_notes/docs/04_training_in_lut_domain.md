# Training in LUT domain (conceptual)

Problem:
A LUT truth table is discrete and non-differentiable.

Solutions used in research-style LUT learning:
1) Learn a real-valued proxy and later discretize
   - e.g., learn a small MLP and then convert to LUT table

2) Use differentiable relaxation:
   - represent LUT outputs as soft probabilities
   - anneal to hard 0/1 outputs

3) Straight-through tricks:
   - like STE for sign, but for LUT selection bits
   - forward uses hard LUT output, backward uses surrogate gradient

Practical approach for our internship roadmap:
- Start with BNN baseline
- Define k-bit groups
- Train small k-input subnetworks with STE
- Convert them into LUT tables by enumerating all 2^k inputs
