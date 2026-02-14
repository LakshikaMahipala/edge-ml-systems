# Metrics definitions (ranking)

We compare methods by ranking candidates.

Core metrics:
- Spearman rank correlation (agreement between two rankings)
- Top-K overlap: how many architectures overlap in top K (K=5,10)

Rank sources:
- Reduced-training proxy rank: score from tiny training budget
- Zero-cost SNIP rank: snip_score
- Zero-cost GradNorm rank: gradnorm_score
- FPGA-aware rank: final_score from acc_proxy - λ * hw_cost
- RL rollouts: reward from controller training logs
