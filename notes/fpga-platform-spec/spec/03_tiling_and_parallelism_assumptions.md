Tiling + parallelism assumptions (so performance claims are meaningful)

We explicitly state what the kernel does in hardware terms.

INT8_FC v1
- OUT parallel accumulators
- IN serial MACs
- cycles_per_vector ≈ IN
- critical path: adder tree avoided (one MAC per cycle)
- scaling knob: unroll IN by U (future), costs more DSPs

DWCONV1D v0
- sequential loops over (c,t,k)
- cycles ≈ C*(L-2)*K
- scaling knob (future):
  - parallelize channels
  - pipeline window shift registers (II=1)

General FPGA knobs
- unroll factor
- pipeline depth
- BRAM vs register storage
- fixed-point scaling policy (SHIFT vs mult+shift)

Why this doc matters
This becomes the “contract” used by our timing model and later by autotuning.
