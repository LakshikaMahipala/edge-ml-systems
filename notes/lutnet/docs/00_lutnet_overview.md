# LUTNet overview 

Core idea:
Instead of implementing a neural layer as multiply-accumulate (DSP-heavy),
we map parts of the network into FPGA LUT fabric.

BNN compatibility:
- binary activations/weights reduce compute to logic operations
- logic operations are naturally implemented in LUTs

LUTNet-style thinking:
- represent a neuron (or small group) as a Boolean function over a limited set of inputs
- implement that Boolean function directly as a LUT truth table
- train the network so that these LUT truth tables produce good accuracy

Why this matters:
- frees DSPs for other tasks or allows more parallelism
- can achieve high throughput under strict resource constraints
