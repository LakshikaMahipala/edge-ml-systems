# Models: regression vs ranking

Regression:
- predict latency_p50_ms directly
- evaluate: MAPE, RMSE, Spearman

Ranking (pairwise):
- learn s(graph) score
- predict ordering by comparing s(A) vs s(B)
- evaluate: accuracy on pairs + Spearman over graphs

Few-shot advantage:
Ranking needs fewer absolute measurements; it only needs relative comparisons.
