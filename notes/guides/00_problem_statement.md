# Mini-project 8: Predictor-based schedule selection

## Problem
Autotuning conv schedules is expensive because measuring each candidate is slow.

We want to:
1) generate many candidate schedules (tiling configs)
2) predict latency for each candidate using a learned predictor
3) measure only the top-k predicted candidates (later)
4) choose the best schedule with far fewer measurements

## Why this matters
This is the standard “learned cost model” idea used by TVM meta-schedule and many NAS systems.

## Extension
We apply the same approach to FPGA kernel design knobs (UNROLL, II):
- treat FPGA design points like schedules
- train predictor on FPGA sweep dataset (estimated now, measured later)
