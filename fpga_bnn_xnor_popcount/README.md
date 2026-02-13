FPGA BNN XNOR-Popcount Dot Product 

What this is
- A synthesizable RTL primitive for binary dot products:
  packed bits -> XNOR -> popcount -> signed accumulation

Math
For each WORD (32 bits):
  matches = popcount( xnor(a_bits, w_bits) )
  contrib = 2*matches - 32
Total acc = sum(contrib across words)

Why this matters
- This is the FPGA-native core of BNN inference.
- Replaces MAC arrays with LUT logic and adder trees.

How to run later (simulation)
1) Compile:
   vlog rtl/popcount32.v rtl/xnor_popcount_dot.v rtl/bnn_dot_top.v sim/tb_bnn_dot_top.v
2) Run:
   vsim tb_bnn_dot_top; run -all

Correctness definition
- acc_out must exactly match Python golden model (python_ref/bnn_golden.py)
- no tolerance: it is bit-exact integer arithmetic.

Next step (Week 12 Day 6/7)
- integrate into a simple “BNN layer” with multiple output neurons
- measure cycles/throughput with counters
