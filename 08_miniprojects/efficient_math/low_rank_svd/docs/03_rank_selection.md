Rank selection strategy

Practical strategy:
- choose r to capture X% of energy:
  energy(r) = sum_{i<=r} σ_i^2 / sum_i σ_i^2

Targets:
- 90% energy: aggressive compression (more error)
- 95% energy: common compromise
- 99% energy: conservative

Hardware-aware twist
Even if energy suggests r=300, the hardware might prefer r in {64,128,256}
because those map cleanly to SIMD/unroll factors.
So we round r to hardware-friendly values.
