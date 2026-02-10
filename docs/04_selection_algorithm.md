# Selection algorithm

Input
- list of candidates C = {c1..cn}
- predictor f(c) -> predicted latency
- budget k (measure only k candidates)

Algorithm
1) For each candidate ci:
   score_i = f(ci)
2) Sort candidates by score_i ascending
3) Select top-k
4) (Later) measure those k, pick best measured

Metrics
- regret@k:
  regret@k = best_true_latency - best_true_latency_among_topk
- hitrate@k:
  whether the globally best true candidate is included in top-k

In our current stage
- we compute regret@k using estimated labels (proxy ground truth)
- later we swap in measured labels.
