# What to measure (minimum viable points)

We need points that map to our op types:

1) "skip":
- either bypass path or memcpy-like path (if relevant)
- measure as "skip_latency_us"

2) conv3:
- representative input/output channels + feature map size
- e.g., Cin=16, Cout=16, H=W=32, K=3

3) conv5:
- same but K=5

Optional extra points:
- change channels (16->24) to see scaling
- include depthwise conv (if you later add it)

Even if the FPGA kernel implementation is simplified,
the goal is ranking agreement, not absolute correctness.
