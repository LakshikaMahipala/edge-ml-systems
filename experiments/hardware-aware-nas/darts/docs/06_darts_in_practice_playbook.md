# DARTS in practice (Week 14 playbook)

This playbook is the “how you actually run DARTS” manual.

---

## 1) What DARTS gives you (and what it does NOT)
DARTS gives:
- a *ranking preference* over ops per edge via α
- a final discrete genotype by argmax(α)

DARTS does NOT give:
- a guaranteed best discrete architecture
- reliable results if you don’t control collapse + discretization gap

So treat DARTS as:
**a fast architecture proposal engine** that must be validated later.

---

## 2) The minimum viable DARTS pipeline
Search stage:
1) define search space ops per edge
2) build mixed-op supernet (contains all ops)
3) bilevel training loop:
   - update weights w on train
   - update architecture α on val
4) log α evolution and detect collapse early
5) discretize genotype: per edge choose argmax(α)

Evaluation stage:
6) rebuild a discrete network (no mixed ops)
7) train from scratch (honest accuracy)
8) measure latency on target (honest latency)

---

## 3) First-order vs second-order (what to use)
First-order:
- cheaper, easier, good for toy runs and fast iteration

Second-order:
- closer to bilevel math, but can be unstable and heavier

Rule:
Start first-order. Only switch if you have evidence the approximation is failing.

---

## 4) How DARTS fails in real life (and how to catch it)
Failure mode A: skip dominance
- symptom: α quickly moves to skip for most edges
- why: skips give good gradient flow early
- mitigations:
  - delay skip availability
  - add drop-path
  - regularize α (entropy penalty)
  - constrain number of skip edges

Failure mode B: discretization gap
- symptom: mixed network is “good” but discrete network trains badly
- why: mixture trained weights do not match single-op weights
- mitigations:
  - retrain discrete from scratch (always)
  - consider one-shot as alternative for ranking
  - use stronger validation protocol (aging evaluation)

Failure mode C: proxy lies (hardware unaware)
- symptom: high-accuracy genotype is unusable on FPGA/edge
- mitigation: add hardware regularizer (Section 5)

---

## 5) Hardware-aware DARTS recipe
Modify α objective:

L_alpha = L_val + λ * ExpectedLatency(α)

Where expected latency is:
Σ softmax(α)_k * latency(op_k)

Two latency sources:
1) Layerwise constant table (fast prototype)
2) FPGA proxy cycles (better directional estimate)

Validation:
- measure 5–10 FPGA kernel points
- check Spearman correlation between proxy vs measurements

---

## 6) What to ship in GitHub
At minimum:
- the supernet code
- training loop
- genotype export
- discrete rebuild code
- benchmark harness
- a report scaffold + templates

You already shipped all of this in Week 14.
