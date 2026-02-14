# Proxy reliability + compute budget notes

This week we built 4 ranking signals:
1) Reduced-training proxy (tiny training)
2) Zero-cost proxies (SNIP-like, GradNorm)
3) RL controller reward (toy REINFORCE)
4) FPGA cost proxy (cycles/LUT/BRAM/BW proxies)

---

## 1) Reduced-training proxy (most “honest” early signal)
Compute cost: medium (needs training steps).
Reliability: higher than zero-cost, but still noisy.

Strengths:
- captures optimization dynamics (learnability)
- usually correlates better with real accuracy than pure init-based scores

Weaknesses:
- mis-ranking if some models learn slowly
- sensitive to dataset subset and seed at low steps
- expensive if candidate pool is large

Use when:
- you can afford some training compute
- you want a practical shortlist

---

## 2) Zero-cost proxies (fast filtering)
Compute cost: very low (one minibatch fwd/bwd).
Reliability: variable; depends on space and dataset.

Strengths:
- extremely fast → good for pruning huge spaces
- easy to average across seeds

Weaknesses:
- can be dominated by init noise
- tends to prefer larger models unless normalized
- batchnorm and saturation can distort gradient-based signals

Use when:
- you need to filter 10k+ candidates quickly
- you will later confirm with reduced-training

---

## 3) RL controller reward (search strategy, not evidence)
Compute cost: low-to-medium depending on evaluation.
Reliability: depends entirely on reward quality.

Strengths:
- focuses exploration on “promising” regions
- reusable framework once reward is real (accuracy + hardware latency)

Weaknesses:
- if reward is proxy-only, it learns proxy quirks
- can collapse to trivial solutions if penalties dominate

Use when:
- you have a stable evaluation function
- you want automated exploration, not manual tuning

---

## 4) FPGA cost proxy (constraint awareness)
Compute cost: tiny (closed-form estimates).
Reliability: good for directional ranking, not for absolute claims.

Strengths:
- forces hardware feasibility early
- prevents picking architectures that are impossible to pipeline/store

Weaknesses:
- doesn’t know real routing/timing
- must be calibrated later with measured kernel points

Use when:
- targeting FPGA/edge acceleration
- you need “hardware-first” candidate selection

---

## Practical rule (intern-grade)
Pipeline:
Zero-cost (fast filter) → Reduced-training (shortlist) → FPGA-aware rank (constraint gate) → Full training + real latency measurement (final selection)
