# Post-search selection protocol

After training supernet:
1) Sample N subnets (e.g., N=200)
2) Evaluate each subnet quickly on validation (shared weights)
3) Rank and pick top-K
4) (optional) re-evaluate top-K multiple seeds/batches to reduce noise
5) export best subnet config (JSON) for retraining from scratch later

Today we implement steps 1–3 and export config.
