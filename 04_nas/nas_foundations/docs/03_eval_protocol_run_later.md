# Evaluation protocol 

We will evaluate each architecture in stages:
1) proxy metrics (fast):
   - params
   - MACs/FLOPs
   - layer count
2) reduced training proxy:
   - few epochs on small subset
3) full training:
   - for top candidates only

For Week 13 Day 1, we implement only (1) and scaffolding for (2).
