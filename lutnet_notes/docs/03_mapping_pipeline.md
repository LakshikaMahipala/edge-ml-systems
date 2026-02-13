# Mapping pipeline (BNN → LUTNet-style implementation)

Pipeline view:

1) Start with a trained BNN or sparse/Binary network.

2) Choose a partition strategy:
   - partition inputs into groups of size k (e.g., 4–6)
   - each group feeds a LUT that outputs 1 bit

3) Replace parts of the network:
   - Instead of computing XNOR+popcount+threshold for those groups,
     implement a LUT truth table for each group.

4) Compose LUT outputs into next stage:
   - either another LUT layer
   - or a reduced popcount stage with fewer inputs

5) Iterate: prune + refit
   - remove groups that contribute little
   - retrain to recover accuracy

Key output:
- a set of LUT truth tables
- wiring specification (which inputs feed which LUT)
- pipeline stage plan for timing
