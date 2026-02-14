# Zero-cost proxies 

Goal:
Score architectures without training.

Common approach:
- initialize model randomly
- run one minibatch
- compute a score based on gradients / saliency / signal propagation

Why it can work:
Architectures that have healthier gradient flow and useful capacity
often show better signals even at initialization.

Why it can fail:
- randomness sensitivity
- some architectures need training dynamics to emerge
- scores may not correlate on hard datasets
