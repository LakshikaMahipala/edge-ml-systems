Reduced-training proxy + aging evaluation 

What this does
- scores candidate architectures using tiny training budget
- periodically re-evaluates top candidates (aging) to reduce mis-ranking

Run later
PYTHONPATH=../nas_foundations/tiny_cnn_search_space/src python src/run_proxy_experiment.py \
  --candidates ../nas_foundations/tiny_cnn_search_space/results/candidates.jsonl \
  --steps 30 --every_k 20 --top_m 5 --extra_steps 60

Outputs
- results/proxy_scores.jsonl
- results/rerank.json (leaderboard snapshots)
