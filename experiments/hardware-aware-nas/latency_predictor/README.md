Simple latency predictors (baseline)

What exists:
1) Regression baseline (MLP) to predict latency_p50_ms
2) Pairwise ranking baseline (few-shot friendly) that learns ordering

Why this matters:
- These baselines validate dataset/feature pipelines.
- Ranking is often what NAS needs.

Run later:
python src/synth_data_generator.py --out data/graph_latency.jsonl --n 200
python src/train_regressor.py --data data/graph_latency.jsonl --out results/regression_results.json
python src/train_ranker.py --data data/graph_latency.jsonl --out results/ranking_results.json
