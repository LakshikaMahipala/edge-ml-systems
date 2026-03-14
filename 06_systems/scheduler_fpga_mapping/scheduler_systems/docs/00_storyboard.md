# storyboard (systems + scheduling)

Goal:
Show that end-to-end performance is a systems problem:
- data movement + scheduling + hardware kernels

Sections:
1) Attachment model: PCIe vs network-attached accelerators
2) Overlay concept: config-driven Conv/BN/ReLU engine
3) Scheduling algorithms: HEFT (heuristic) vs OPT/MILP (exact baseline)
4) FPGA kernel mapping: how measured/proxy kernel points become scheduling inputs
5) Results tables: makespan comparisons and transfer fraction budgeting
6) What must be measured later (FPGA kernel points) and how we update the report
