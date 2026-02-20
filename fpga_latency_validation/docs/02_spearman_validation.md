# Validation method: Spearman rank correlation

We care about ranking, not absolute latency.

Compute:
- predicted latency for each kernel point
- measured latency (p50 or p99)
Then compare rankings with Spearman correlation.

Interpretation:
- Spearman near 1.0: proxy preserves ordering well
- near 0: proxy is random
- negative: proxy is misleading
