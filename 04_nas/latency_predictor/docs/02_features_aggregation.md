# Feature aggregation (graph → vector)

We turn a graph into a fixed-length vector by aggregating operator stats:

Examples:
- total_macs
- total_params
- total_bytes_moved
- count_conv3, count_conv5, count_skip
- depth (node count)
- mean kernel size
- max channels
- arithmetic intensity proxy = macs / bytes

This is surprisingly strong as a latency baseline.
