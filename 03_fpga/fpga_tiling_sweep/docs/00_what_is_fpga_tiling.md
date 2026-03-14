FPGA “tiling” means something slightly different than CPU/GPU tiling.

On FPGA, tiling usually implies:
- how you partition tensors into blocks that fit BRAM
- how many MAC lanes you unroll (parallelism)
- how deeply you pipeline (initiation interval II)
- how you stream data (overlap I/O and compute)

A good FPGA design picks parameters that:
- fit in BRAM/DSP budgets
- sustain high throughput (II=1 ideal)
- minimize external bandwidth needs

In our project, “tiling sweep” = sweep design knobs like:
- unroll factor (parallel MACs)
- tile sizes (block dimensions)
- assumed II
and estimate latency/resource proxies.
