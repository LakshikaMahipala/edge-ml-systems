One-shot / Supernet NAS (minimal)

What exists:
- Supernet with switchable kernel size and width per block
- Uniform subnet sampler
- Weight-sharing training loop
- Post-search selection: evaluate N subnets using shared weights and export best config

Important warning:
Shared-weight accuracy is biased. Always retrain selected subnet from scratch later.

Run later:
python src/trainer.py --steps 300
python src/eval_subnets.py --n 100
python src/export_subnet.py
