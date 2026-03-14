# Design choices and tradeoffs

## Choose k (LUT input size)
- k small (4–6):
  + LUT mapping is easy
  + routing manageable
  - more layers needed to represent complex functions

## Use DSP vs LUT balance
- LUTNet is attractive when:
  - DSP budget is tight
  - logic is abundant
  - operations are binary

## Structured pruning synergy
- structured pruning reduces fan-in
- lower fan-in makes LUT mapping more feasible

## Accuracy vs timing
- aggressive LUT replacement may drop accuracy
- recover by:
  - increasing width
  - adding batchnorm-style normalization
  - keeping first/last layers non-binary
