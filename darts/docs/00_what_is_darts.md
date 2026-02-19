# DARTS (Differentiable Architecture Search) — what it really is

DARTS turns architecture search from a discrete problem into a differentiable one.

Instead of choosing a single operation (e.g., 3x3 conv OR 5x5 conv),
we build a **mixed operation** that contains all candidate ops in parallel,
and we learn **continuous weights** (architecture parameters) that decide how much
each op contributes.

Key objects:
- **w** = normal network weights (filters, linear weights, etc.)
- **α** = architecture parameters (which ops/edges are preferred)

At the end, we discretize:
- for each mixed op, pick the op with the highest α weight

DARTS is attractive because:
- it uses gradient descent instead of expensive search loops
- it feels like "NAS but as training"

But DARTS is also fragile because:
- the relaxation is biased
- optimization is bilevel (train vs validation split)
- it can collapse to cheap/degenerate ops without constraints
