# Joint search space (A × H)

We represent each candidate as:

candidate = {
  "arch": {...},
  "hw": {...}
}

arch space:
- block ∈ {conv3, conv5}
- width ∈ {16, 24}
- depth ∈ {2, 3}

hw space:
- P (parallel MAC lanes) ∈ {8,16,32,64}
- tile ∈ {8,16,32}
- II_factor ∈ {1.0, 1.2}

Total combinations = small enough for exhaustive search in toy setting,
but the structure generalizes to large co-design.
