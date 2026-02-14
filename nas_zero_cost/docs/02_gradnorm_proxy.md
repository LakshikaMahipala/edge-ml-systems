# GradNorm proxy

Idea:
Measure gradient norm magnitude as a proxy for learnability.

One simple version:
score = || ∇_W L ||_2 summed over layers or total.

Interpretation:
If gradients are near-zero at init, training may be difficult.
If gradients are extremely large, training may be unstable.
But in many settings, higher (reasonable) gradient norms correlate with better early learning.
