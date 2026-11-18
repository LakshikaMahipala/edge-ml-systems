# Search space design rules

A good search space is:
- expressive enough to include good models
- constrained enough to search efficiently

Rules:
1) Use a fixed macro-skeleton (stages), search micro choices (blocks).
2) Keep choices discrete and small early (2–5 options each).
3) Include constraint-friendly knobs:
   - channel multipliers
   - depth per stage
   - kernel size (3/5)
   - expansion ratio
4) Avoid too many graph topologies early (skip connections limited).
5) Make architecture encoding simple and stable (JSON serializable).
