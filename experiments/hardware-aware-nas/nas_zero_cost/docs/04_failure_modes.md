# Failure modes (important)

Zero-cost proxies can fail when:
- architecture scores are dominated by random init noise
- dataset is too simple (all look similar)
- proxy prefers large models unless normalized
- batchnorm behavior at init distorts gradients
- activation saturation collapses gradients

Mitigations:
- average proxy over multiple random seeds
- normalize by parameter count
- use consistent init schemes
- log proxy components for debugging
