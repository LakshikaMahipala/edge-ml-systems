FPGA cost model integration

What we use
- fpga_cost_model/src/fpga_latency_estimator_v0.py

What the compiler provides to the cost model
- op type
- shape parameters (IN/OUT or C/L/K)
- quant params (SHIFT)
- payload sizes (implicitly computed in estimator)

What the cost model returns
- T_io (UART model)
- T_compute (cycle model)
- T_total = T_io + T_compute + host

Key point
Even if compute is fast, UART I/O dominates.
So any compiler pass claiming “speedup” must be evaluated end-to-end.
