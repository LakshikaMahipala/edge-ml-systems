Sweep parameters (v0)

We sweep two kernel families:
A) INT8_FC (streaming MAC)
- IN: input dimension
- OUT: output dimension
- UNROLL: MAC lanes (parallelism)
- II: initiation interval (assumed)
- f_clk_mhz

B) INT8_DWCONV1D_K3
- C: channels
- L: length
- K: kernel size (fixed 3)
- UNROLL_C: channels processed in parallel
- II: assumed

Constraints (v0)
- UNROLL must be power-of-two in {1,2,4,8,16}
- UNROLL <= IN for FC
- UNROLL_C <= C for DWConv
- II in {1,2,4} (1 best, 4 worst)

Outputs
- cycles_est
- bytes_total
- latency_est (UART model)
- resource_proxy (rough DSP proxy)
