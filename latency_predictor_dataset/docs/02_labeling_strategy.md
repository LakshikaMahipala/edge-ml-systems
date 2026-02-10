Labeling strategy (v0)

We cannot measure latency yet. So we label with:
- fpga_cost_model estimator (UART + cycles) to produce y_fpga_est_* targets.

Later, we will add measured labels:
- run each config on CPU/GPU/FPGA and fill y_measured columns.

Two-stage use
Stage A (now): learn predictor on estimated labels
- purpose: verify pipeline and feature usefulness

Stage B (later): learn predictor on measured labels
- purpose: real accuracy

Why v0 still matters
Predictor training code, dataset structure, and evaluation are identical.
Only labels change.
