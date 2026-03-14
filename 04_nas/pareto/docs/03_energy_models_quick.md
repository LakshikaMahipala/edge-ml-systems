# Energy models (quick and practical)

Energy is harder than latency because you need power.

Common approximations:
1) Energy ~ Latency * Power
2) Power proxy ~ MACs * switching_activity + memory_access_cost
3) If you can't measure:
   use "energy proxy" = a*MACs + b*bytes_moved

For FPGA/edge work early-stage:
- ranking and tradeoff reasoning matter more than perfect Joules
- later you calibrate with real measurements
