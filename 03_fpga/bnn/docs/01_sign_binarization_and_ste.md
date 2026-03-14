# Sign binarization + Straight-Through Estimator (STE)

Binarization:
b = sign(x) where sign(x) ∈ {+1, -1}

Problem:
sign(.) is non-differentiable → gradients vanish.

Solution:
Use STE:
- forward: b = sign(x)
- backward: pretend it's identity in a range
  db/dx ≈ 1 if |x| <= 1 else 0

This lets training proceed while still forcing binary values in forward.
