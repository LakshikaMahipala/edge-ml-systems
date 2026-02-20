# What is the graph?

We represent a model as a directed graph G=(V,E):

Nodes V:
- each node is an operator (Conv, BN, ReLU, Add, MatMul, etc.)

Edges E:
- tensor flow: output of op u goes into op v

This graph is what frameworks already have (ONNX graph, Torch FX graph).

Important:
Latency is not just sum of node costs due to:
- parallelism
- fusion
- memory effects

But a GNN can learn those patterns if features are good.
