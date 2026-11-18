# Mini-project 11: NAS Baselines + FPGA-aware ranking 

## 1. Overview
This mini-project compares NAS candidate rankings produced by:
- reduced-training proxies
- zero-cost proxies (SNIP-like, GradNorm)
- FPGA cost proxy (cycles/LUT/BRAM/BW)
- RL controller rollouts (toy)

Goal:
Demonstrate why hardware-aware scoring changes which architectures we should pick.

## 2. Search space 
We define a small mobile-style search space with:
- stage depth choices
- out_channels choices
- conv vs MBConv blocks
- kernel size, expansion, SE
Architecture encoding is JSON-serializable.

## 3. RL controller 
A toy REINFORCE controller samples architectures and updates logits to maximize reward.
Current reward uses proxy accuracy + constraints (no full training yet).

## 4. Reduced-training proxy 
We score each architecture after a very small training budget.
This provides a noisy approximation of “learnability”.

## 5. Zero-cost proxies 
We compute:
- SNIP-like saliency: sum |w * grad|
- GradNorm: ||∇W L||₂
Both computed from one minibatch at initialization.

## 6. FPGA cost proxy 
We estimate “hardware pain” as proxy terms:
- cycles_proxy ~ log(ops/P)
- lut_proxy, bram_proxy, bw_proxy (heuristics)
Final FPGA-aware score:
final = acc_proxy - λ * (cycles + lut + bram + bw)

Honesty note:
These are proxies for ranking, not synthesis results.

## 7. Rank comparison
We compare rankings using:
- Spearman rank correlation
- Top-K overlap (K=5,10)

Results tables:
- report/tables/template_rank_comparison.csv (to be populated after runs)

## 8. Conclusions (expected)
- zero-cost and reduced-training often disagree (different signals)
- FPGA-aware ranking shifts candidates toward smaller/channel-efficient blocks
- when all ranks agree, those candidates are strong “safe picks”

## 9. Run-later instructions
See docs/01_run_later_plan.md
