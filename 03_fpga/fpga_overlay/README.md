FPGA overlay for Conv/BN/ReLU (ISA + tiling)

What this module delivers:
- Overlay architecture description
- Layer descriptor ISA (JSON contract)
- Memory map and tiling/dataflow plan
- Cycle model proxy stub

Why it matters:
This overlay is the bridge between ML models and reusable FPGA execution.

Next:
Week 16 Day 3–4: scheduling (HEFT/MILP) will treat overlay kernels as tasks with predicted latency.
