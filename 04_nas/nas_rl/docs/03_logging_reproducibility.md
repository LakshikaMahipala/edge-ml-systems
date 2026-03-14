# Logging and reproducibility

We log every rollout as JSONL:
- seed
- step
- sampled architecture encoding
- reward components (acc_proxy, params, macs)
- total reward

This is essential because NAS is stochastic.
We must be able to reproduce a run exactly via seed.
