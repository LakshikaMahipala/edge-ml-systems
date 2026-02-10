Feature engineering (what the predictor will actually learn)

We want device-agnostic, meaningful features:
- macs (compute intensity)
- bytes_in + bytes_out (data movement)
- arithmetic_intensity = macs / bytes_total
- shape ratios (Cin/Cout, etc.)
- log features (log(macs), log(bytes))

Why log features help
Latency often scales roughly linearly in log-space across orders of magnitude.
A simple model learns better.

Categoricals
- op_type is categorical -> one-hot encoding (later)
- interface is categorical -> one-hot

Important
If labels are UART-dominated, macs won’t matter.
So we keep both macs and bytes features to let the model discover regimes.
