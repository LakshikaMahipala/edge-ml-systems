# Week 16 resume bullets

- Built an end-to-end accelerator latency budgeting framework comparing PCIe-attached vs network-attached accelerators; quantified transfer-dominated regimes via copy-cost modeling.
- Designed an FPGA Conv/BN/ReLU overlay concept with layer-descriptor ISA (JSON contract), memory map, tiling/dataflow strategy, and proxy cycle model for performance estimation.
- Implemented heterogeneous DAG scheduling using HEFT for CPU/GPU/FPGA and added an exact optimal baseline (brute-force OPT as MILP proxy) to quantify heuristic gap on small instances.
- Created a latency registry and pipeline→DAG generator to map FPGA kernel latencies into scheduling experiments, enabling reproducible makespan and placement studies.
