# Logging + debugging DARTS (what you must record)

DARTS is fragile. If you don’t log the right things, you will not know why it failed.

---

## 1) Log α distributions over time
For each mixed op edge:
- softmax(α) probabilities
- argmax op
- entropy of softmax(α)

Why:
- entropy collapsing too early means search is degenerating.

---

## 2) Log validation loss vs latency penalty separately
For latency-aware DARTS:
- val_loss
- latency_total
- alpha_objective = val_loss + λ*latency

Why:
- if λ is too large, latency dominates and accuracy collapses
- if λ is too small, architecture ignores latency

---

## 3) Collapse detectors (simple rules)
- if >70% edges choose "skip" for >N steps → skip dominance
- if entropy < 0.1 too early → premature collapse
- if genotype keeps oscillating every log interval → unstable α updates

---

## 4) Reproducibility log
Always record:
- seed
- train/val split strategy
- steps
- learning rates (w and α)
- λ (latency weight)
- device + torch version

This turns “toy result” into “paper-grade traceability”.
