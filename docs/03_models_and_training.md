# Models and training

Baseline: HistGradientBoostingRegressor
- strong tabular learner
- fast training
- interpretable importance (later)

Neural: MLP regressor
- needs scaling
- may generalize better with enough data

Training target
- log(1 + latency_ms)

Why log?
Schedule latencies span orders of magnitude.
Log stabilizes regression.

Validation
- group splits for config leakage prevention
- rank correlation is mandatory
