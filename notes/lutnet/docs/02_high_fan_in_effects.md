# High fan-in effects (why k matters)

Fan-in = number of inputs feeding into a logic function/neuron.

BNN dot product has high fan-in:
- one output neuron depends on many input bits (often 64–1024)

But FPGA LUT has limited fan-in:
- e.g., k=6 inputs

Therefore:
- you cannot map a full neuron directly into one LUT
- you must either:
  (A) decompose into LUT networks (trees of LUTs), OR
  (B) train small fan-in subfunctions that combine hierarchically

Why high fan-in matters:
- larger fan-in increases logic depth and routing complexity
- deeper logic increases latency and reduces max clock
- routing dominates area and timing when fan-in is large

The LUTNet insight:
- encourage networks where useful computation happens in small k-input groups,
  then compose them across layers.
