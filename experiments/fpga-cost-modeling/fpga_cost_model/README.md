FPGA Cost Model 

What this is
A static FPGA latency estimator v0:
T_total = T_io(UART) + T_compute(cycles/f_clk) + T_host

Supports
- INT8_FC
- INT8_DWCONV1D_K3
- MobileNet-like block v0 = DWConv + ReLU

Run later
python src/fpga_latency_estimator_v0.py --graph_json examples/example_block_mobilenet_v0.json --out outputs/block_est.json

Key insight
Under UART, T_io dominates.
This tool quantifies that and gives a reusable interface for HW-aware search later.
