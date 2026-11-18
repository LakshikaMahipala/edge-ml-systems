Latency-aware DARTS

What this module adds:
- Expected latency regularizer for DARTS MixedOps:
  E[latency] = sum softmax(alpha)_k * latency(op_k)
- Alpha update uses:
  loss_alpha = val_loss + lam * expected_latency

Run later:
PYTHONPATH=../darts_impl/src python src/trainer_latency_aware.py --steps 200 --lam 0.2

Outputs:
- results/alpha_latency_history.jsonl
- results/best_genotype_latency.json

Replace later:
- latency_table proxy costs -> real microbench latencies or FPGA estimator
