Search space + cost model

Search space
A schedule search space is the set of all legal schedule parameterizations.
For conv2d, common knobs:
- tile_h, tile_w, tile_c
- vectorize width
- unroll factors
- parallelization strategy

Cost model
Because measuring every candidate is too slow, TVM uses a cost model:
- learns mapping from schedule features -> predicted latency
- guides the search toward promising areas

Two extremes
- Pure random search: simple, but inefficient
- Pure model-based search: efficient if cost model is good

Why we care for FPGA later
Our FPGA latency estimator plays the same role:
- given an implementation choice, score it
- guide search / co-design loops
