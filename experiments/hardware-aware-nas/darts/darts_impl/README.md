Minimal DARTS (first-order)

What this implements:
- MixedOp with learnable alpha logits over {skip, conv3, conv5}
- Tiny cell DAG with 3 mixed edges
- Supernet (stack cells) for toy classification
- First-order DARTS training loop:
  - update weights on train batch
  - update alphas on val batch
- Genotype extraction to JSON

Run later:
python src/darts_trainer.py --steps 200

Outputs:
- results/alpha_history.jsonl
- results/genotype.json
