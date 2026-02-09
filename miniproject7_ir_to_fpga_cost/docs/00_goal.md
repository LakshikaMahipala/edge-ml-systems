Mini-project 7: From IR to FPGA cost 

Goal
Demonstrate the compiler+hardware loop:
1) Start from Relay IR text (exported from a model)
2) Run an analysis pass (op counting)
3) Run a rewrite pass (toy fusion rewrite)
4) Convert a subset of the IR into a target graph format
5) Run FPGA latency estimator v0

Why it matters
This is the workflow of HW-aware compilation:
- you analyze the model graph
- you rewrite for efficiency or hardware constraints
- you score candidates with a cost model
