# Sampling strategies

We need a distribution over subnets.

Simplest:
- uniform sampling over each choice dimension

Examples:
- choose kernel size ∈ {3,5} uniformly
- choose width multiplier ∈ {0.5, 1.0} uniformly

Important:
Sampling distribution affects which subnets get trained well.
So you must log sampling stats.
