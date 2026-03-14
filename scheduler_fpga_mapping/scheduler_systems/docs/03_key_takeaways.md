# Key takeaways

1) If transfer dominates, faster kernels do not improve end-to-end latency (Amdahl).
2) Overlay design is a software contract: descriptor ISA + tiling rules enable reuse.
3) HEFT is a strong heuristic for heterogeneous scheduling with low compute overhead.
4) OPT/MILP baselines are crucial for understanding how far a heuristic is from optimal.
5) FPGA kernel points must be integrated into scheduling to make realistic system claims.
