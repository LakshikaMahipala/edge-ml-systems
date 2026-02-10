Metrics (why each matters)

MAE (ms)
- average absolute error in milliseconds
- intuitive for latency

RMSE (ms)
- penalizes large mistakes more strongly than MAE

MAPE (%)
- relative error, but unstable when true latency is near 0
- still useful for comparing across scales

Spearman rank correlation (ρ)
- measures whether the predictor preserves ordering
- extremely important for:
  - NAS ranking
  - schedule selection
Even if absolute error is not perfect, high rank correlation can still enable good search.

Rule of thumb
- Good enough for search: Spearman ρ > 0.8
- Strong predictor: ρ > 0.9
