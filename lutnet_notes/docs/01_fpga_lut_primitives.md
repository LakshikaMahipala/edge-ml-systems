# FPGA LUT primitives (what we can actually build)

LUT = small SRAM-like truth table:
- k-input LUT can implement any Boolean function of k bits
- typical k is 6 (6-input LUT) in modern FPGAs

Interpretation:
- a 6-LUT maps 6 input bits → 1 output bit
- the LUT configuration bits define the truth table

How BNN layers map:
- XNOR gates: in LUTs
- popcount: adder trees (LUTs + carry chains)
- threshold compare: LUTs + carry

How LUTNet extends this:
- replace "XNOR+popcount+threshold" subgraphs with a single LUT function
  when the fan-in (k) is small enough.
