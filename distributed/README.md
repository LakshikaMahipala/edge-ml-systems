Distributed 

Goal
Understand ring allreduce (used in distributed training) by implementing a CPU-only simulator.

What’s included
- docs/ring_allreduce_theory.md
- scripts/ring_reduce_scatter_only.py
- scripts/ring_allreduce_sim.py

Run later
python scripts/ring_reduce_scatter_only.py --N 4 --L 1024
python scripts/ring_allreduce_sim.py --N 4 --L 1024
python scripts/ring_allreduce_sim.py --N 8 --L 8192 --avg
