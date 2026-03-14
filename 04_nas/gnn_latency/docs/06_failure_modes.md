# Failure modes (what breaks latency predictors)

1) Operator fusion changes graph semantics
- predictor trained on unfused graphs but runtime fuses ops → mismatch

2) Memory effects dominate
- cache, bandwidth, alignment effects not captured by MACs

3) Device mode changes
- clocks, DVFS, power caps change latency

4) Out-of-distribution shapes
- predictor fails if trained only on small shapes

Mitigation:
- include runtime/device flags
- include bytes moved + intensity features
- collect calibration measurements periodically (aging evaluation)
