# FPGA kernel evidence (BNN primitive)

Implemented:
- popcount32.v (adder tree)
- xnor_popcount_dot.v (streaming packed words, signed accumulation)
- bnn_dot_top.v (wrapper)
- tb_bnn_dot_top.v (known-vector test)

Golden reference:
- python_ref/bnn_golden.py

Correctness:
- exact integer match between RTL acc_out and python dot_acc

What remains to run later:
- simulation of tb
- randomized regression vectors
- cycle counting and throughput measurement
