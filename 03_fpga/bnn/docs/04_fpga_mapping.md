# FPGA mapping of BNN

BNN compute on FPGA:
1) store weights as bit-packed BRAM words
2) stream activation bits
3) compute XNOR in LUTs
4) popcount using adder trees (can be pipelined)
5) optional scaling + bias
6) activation sign for next layer

What dominates:
- memory bandwidth for bitstreams
- popcount adder-tree depth (pipeline II)
- accumulation and thresholding

BNN is often the most FPGA-native NN style.
