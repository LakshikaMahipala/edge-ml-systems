# One-shot / Supernet NAS 

Supernet:
A single over-parameterized network that contains many candidate sub-networks (subnets).

One-shot NAS workflow:
1) Train the supernet with weight sharing.
2) Sample many subnets.
3) Evaluate subnets quickly using the shared weights.
4) Select a “best” subnet and optionally retrain it from scratch.

Key difference from DARTS:
- DARTS uses soft mixtures and α gradients.
- One-shot uses discrete subnet sampling, but reuses shared weights.

Why it is useful:
- avoids training each candidate separately (huge savings)

Why it is dangerous:
- weight sharing creates biased evaluation (ranking errors).
