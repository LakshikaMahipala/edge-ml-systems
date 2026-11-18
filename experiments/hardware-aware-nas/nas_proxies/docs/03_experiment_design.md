# Experiment design (run later)

We will:
1) Load candidate architectures from JSONL (from Day 1 sampler).
2) Score each with a tiny training proxy budget.
3) Keep a leaderboard.
4) Every K candidates scored:
   - re-evaluate top-M (aging)
   - compare rank changes

Outputs:
- proxy_scores.jsonl: raw scores per run
- rerank.json: leaderboard snapshots before/after aging

Later:
- compute correlation between proxy rank and final rank (if full training added).
