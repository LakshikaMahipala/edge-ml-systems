# Run-later commands

1) Generate synthetic dataset:
python src/synth_data_generator.py --out data/graph_latency.jsonl --n 200

2) Train regressor:
python src/train_regressor.py --data data/graph_latency.jsonl --out results/regression_results.json

3) Build pairs + train ranker:
python src/train_ranker.py --data data/graph_latency.jsonl --out results/ranking_results.json
