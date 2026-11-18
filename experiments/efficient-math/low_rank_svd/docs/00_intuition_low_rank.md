Low-rank intuition 

Dense linear layer
y = x W + b
W is (in_features x out_features)

If W has redundancy (correlated features), it can be approximated by low rank r:
W ≈ A B
A: (in x r)
B: (r x out)

Compute becomes:
xW ≈ (xA)B
Two smaller matmuls.

Why this is valuable
- reduces parameters
- reduces MACs
- enables smaller hardware datapaths (fewer DSP lanes)

Hidden truth
Low-rank works best when:
- layer is large
- model has redundancy
- chosen rank is small enough
Otherwise, overhead can cancel benefits.
