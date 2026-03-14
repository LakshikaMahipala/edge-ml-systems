Offline filter transform (why it matters)

In inference, weights are constant.
Winograd lets you precompute the transformed filter U:
- reduces runtime work
- reduces total multiplications further
- makes inference more like: "transform input + elementwise multiply"

Practical pipeline
- during model export / compile: transform filters once
- store transformed filters in memory (different layout)
- runtime uses transformed filters directly

Cost tradeoff
- you store more intermediate values (larger weight footprint)
- but you reduce compute per output
