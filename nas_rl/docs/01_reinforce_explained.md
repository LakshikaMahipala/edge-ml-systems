# REINFORCE (policy gradient) in plain terms

We want to maximize expected reward:
J(θ) = E_{a ~ πθ}[ R(a) ]

Gradient:
∇θ J(θ) = E[ R(a) * ∇θ log πθ(a) ]

Meaning:
- if reward is high, increase probability of sampled actions
- if reward is low, decrease probability

Variance reduction:
Use baseline b:
(R(a) - b) instead of R(a)
Common b = running mean reward
