# Reward design under constraints

A common pattern:
reward = accuracy_term - λ1 * latency_term - λ2 * size_term

In our toy setup (no real training yet):
- use proxy accuracy (stub)
- use params and MACs as constraints proxies

Important:
Reward scaling matters.
If MAC penalty dominates, controller will collapse to tiny models.
We normalize terms and tune λ.
