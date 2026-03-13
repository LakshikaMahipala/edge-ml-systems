# Estimation ladder for toy co-design

Latency estimation ladder:
L0: MACs proxy
L1: FPGA cost proxy model (Week 13 Day 5, cycles_proxy)
L2: measured kernel points (Week 14 Day 5 validation scaffold)

Accuracy estimation ladder:
A0: capacity proxy (ops/params)
A1: reduced-training proxy (Week 13 Day 3 harness)
A2: full training accuracy (later)

Principle:
Use cheap estimators for broad search,
use expensive measurements to calibrate and finalize.
