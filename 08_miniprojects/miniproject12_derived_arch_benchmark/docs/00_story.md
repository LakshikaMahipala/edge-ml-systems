# Mini-project 12 story

Goal:
Show end-to-end: Search → Derive discrete architecture → Benchmark.

Inputs:
- DARTS genotype JSON (from darts_impl or darts_latency_aware)
- One-shot best_subnet.json (from oneshot_supernet)

Outputs:
- Discrete DARTS model
- Discrete One-shot model
- Baseline small CNN
- Benchmark harness (latency p50/p99 + accuracy placeholder)

This is the key deliverable that turns "NAS theory" into "real engineering artifact".
