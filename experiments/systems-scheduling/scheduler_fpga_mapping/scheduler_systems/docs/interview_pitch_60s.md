# Week 16 — 60-second interview pitch

I built a systems-level framework to reason about end-to-end accelerator performance, not just kernel speed.
First, I modeled transfer overheads and compared PCIe-attached versus network-attached accelerators using a copy-cost budget calculator to identify transfer-dominated regimes.

Then I designed an FPGA Conv/BN/ReLU overlay concept with a layer-descriptor ISA, memory map, and tiling/dataflow plan, plus a proxy cycle model to estimate compute-versus-memory bounded performance.

On the systems side, I implemented HEFT scheduling for heterogeneous CPU/GPU/FPGA DAGs and added an exact optimal baseline using brute-force enumeration for small graphs, allowing me to quantify the HEFT-vs-optimal gap.

Finally, I built a latency registry and a pipeline→DAG generator that injects per-device kernel latencies so the scheduler can choose placements and estimate makespan. The repo includes an evidence policy and a run-later plan to calibrate proxies with real FPGA kernel measurements.
