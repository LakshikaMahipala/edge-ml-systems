# Expected results

If predictor is good:
- Spearman > 0.8
- regret@10 small (top-10 includes near-best candidates)
- top-k selection reduces measurement count dramatically

FPGA note
Under UART, most candidates differ mainly in compute, but I/O dominates.
So:
- predictor may learn that UNROLL/II barely changes total latency (true under UART)
This is not a failure — it's the correct systems conclusion.
