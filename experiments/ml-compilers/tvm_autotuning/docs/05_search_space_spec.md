Conv2D tiling search space spec (v0)

Workload abstraction
We tune a representative conv2d-like loop nest.
We focus on tiling along:
- output channels (Cout)
- input channels (Cin / reduction)
- spatial (H, W)

Knobs (CPU target v0)
- tile_co: {4, 8, 16, 32, 64}
- tile_ci: {4, 8, 16, 32, 64}
- tile_y:  {1, 2, 4, 7, 14}
- tile_x:  {1, 2, 4, 7, 14}
- vec:     {1, 4, 8, 16}
- unroll:  {0, 1}  (0=no unroll, 1=unroll inner)

Constraints
- tile_co divides Cout (or last tile is remainder)
- tile_ci divides Cin (or remainder)
- vec must divide tile_co (typical for vectorization along channels)
- avoid tiny tiles: tile_y*tile_x >= 4 (heuristic)

Proxy cost outputs
- estimated MACs (fixed for shape)
- estimated bytes moved (approx)
- proxy score = bytes/MAC (lower is better) or MAC/bytes (higher is better)

Purpose
This spec defines the space we will later feed into real TVM schedules.
