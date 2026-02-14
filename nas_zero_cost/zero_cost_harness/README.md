Zero-cost proxies 

Implemented proxies
- SNIP-like saliency: sum |w * grad|
- GradNorm: L2 norm of gradients

Run later
PYTHONPATH=../nas_foundations/tiny_cnn_search_space/src python src/run_zero_cost.py --max 100
python src/correlate.py --zero_cost results/zero_cost.jsonl --proxy_scores ../nas_proxies/reduced_training_harness/results/proxy_scores.jsonl

Outputs
- results/zero_cost.jsonl
- results/correlation.json
