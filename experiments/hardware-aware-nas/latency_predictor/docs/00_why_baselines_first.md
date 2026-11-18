# Why baselines first 

Before building a full GNN latency predictor, we always ship a baseline.

Reason:
- Baselines catch dataset bugs, labeling bugs, and leakage early.
- If a baseline cannot learn anything, a GNN won’t magically fix it.

We implement two baselines:
1) Regression: predict latency value (p50)
2) Pairwise ranking: predict "A faster than B"

Ranking is often better for few-shot NAS.
