# CoD-style workflow pattern (CoD-1 / CoD-2 family)

Generic co-design loop:

1) Define joint search space:
   - architecture knobs A
   - hardware knobs H

2) Define objective:
   - accuracy(A) and cost(A,H) where cost includes latency/energy/area

3) Use an estimation ladder:
   L0: cheap proxies (MACs, param count)
   L1: learned latency predictor / FPGA proxy
   L2: occasional real measurement (synthesis or on-board timing)

4) Search algorithm:
   - evolutionary search / Bayesian opt / RL
   - use predictor to avoid expensive evaluation
   - periodically calibrate predictors with real measurements

5) Output:
   - Pareto frontier across (accuracy, latency, energy)
   - final recommended design points with evidence
