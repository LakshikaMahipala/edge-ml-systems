# Using FPGA proxy as a latency estimator (conceptually)

We already built an FPGA proxy model that outputs:
cycles_proxy, lut_proxy, bram_proxy, bw_proxy

For a latency-aware objective, the most direct term is cycles_proxy.

So we can treat:
latency_proxy = cycles_proxy

Later :
we validate whether this proxy agrees with measured FPGA kernel points.

For now:
we integrate it as a drop-in function that returns a differentiable expected latency.
