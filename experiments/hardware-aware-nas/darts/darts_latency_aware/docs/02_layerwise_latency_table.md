# Layerwise latency table

We assign each op a base latency cost:
- skip: very small
- conv3: medium
- conv5: larger

We can use:
- measured microbenchmarks later
- or proxy constants for now

We keep the table explicit in code so it can be replaced by real measurements.
