TVM autotuning: core concepts 

What autotuning is
Autotuning is automated search over implementation choices (“schedules”).
For a single operator like conv2d, there are many ways to implement it:
- tile sizes
- loop ordering
- vectorization
- unrolling
- memory locality choices
Different choices produce very different latency.

Why it works
- Hardware performance is dominated by low-level details.
- Humans can’t reliably guess best tiling for every shape/device.
- Autotuning measures candidates and learns what works.

Typical autotuning loop
1) Define search space (possible schedules)
2) Propose candidate schedules (search strategy)
3) Generate code for each candidate
4) Measure on target hardware
5) Update cost model / policy
6) Repeat, then choose best

Key outputs
- Best schedule config
- Best measured latency
- Search trace (for reproducibility)

Important caveats
- Measurements must be stable (warmup, pin CPU freq, isolated machine)
- Search space size matters (too large = slow search)
- Cost model quality matters when measurements are expensive
