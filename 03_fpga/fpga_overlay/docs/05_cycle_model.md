# Cycle model (proxy)

We need a simple performance model for scheduling and design-space exploration.

For one conv tile:
Ops_tile ≈ Tm * Tr * Tc * Tn * K*K

If we have P parallel MACs:
cycles_compute ≈ Ops_tile / P

Total cycles includes:
- compute cycles
- load/store cycles (bandwidth limited)
- overhead (loop/controller)

So:
cycles_tile ≈ max(cycles_compute, cycles_mem) + overhead

This model is used in:
- NAS cost proxies
- scheduling simulators (Week 16 Day 4–5)
