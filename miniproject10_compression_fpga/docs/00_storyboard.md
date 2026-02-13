# Mini-project 10 storyboard 

Goal:
Show a complete compression pipeline and connect it to FPGA-realistic execution.

Chapters:
1) Quantization
   - PTQ vs QAT concepts
   - our QAT toy project structure

2) Pruning
   - unstructured vs structured
   - why structured matters for speed

3) BNN
   - why BNN is hardware-first
   - STE + accuracy caveats

4) LUTNet direction
   - LUT mapping idea and high fan-in limitation

5) FPGA evidence
   - implemented XNOR-popcount dot kernel in RTL
   - Python golden model for bit-exact verification
   - plan to measure cycles/throughput later
