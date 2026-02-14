# Wrap (NAS fundamentals → FPGA-aware ranking)

What we built:
- Day 1: search space + architecture encoding + model builder + MAC/params proxy
- Day 2: toy RL controller (REINFORCE) + JSONL rollouts
- Day 3: reduced-training proxy harness + aging evaluation
- Day 4: zero-cost proxies (SNIP-like, GradNorm) + Spearman correlation script
- Day 5: FPGA cost proxy model + FPGA-aware ranking + join script
- Day 6: Mini-project 11 report scaffold (rank comparison)
- Day 7: reliability + compute budget consolidation (this doc)

What is intentionally missing (run later):
- true dataset training
- measured latency on target hardware
- calibration of FPGA cost proxy against real kernel points

Next week direction:
- DARTS / differentiable NAS (bi-level optimization)
- supernet/one-shot NAS and weight sharing reliability
