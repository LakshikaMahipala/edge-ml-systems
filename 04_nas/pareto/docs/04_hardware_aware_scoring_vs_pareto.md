# Why Pareto is better than single weighted scoring

Latency-aware DARTS used:
score = val_loss + λ * latency

This is a weighted-sum scalarization.

Problem:
Different λ values trace different tradeoffs, but one λ hides the full frontier.

Pareto evaluation:
- keeps the entire tradeoff set visible
- lets you pick later based on constraints (latency budget, energy cap)

Rule:
Use weighted scores for training/search convenience,
but report Pareto front for honest engineering conclusions.
