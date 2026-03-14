Split protocol (train/val/test)

We do NOT randomly split rows naïvely if shapes repeat.
We split by configuration groups to prevent leakage.

Protocol v0
- Define a config_id hash from (op_type + key shape params + quant)
- Group by config_id
- Split config_ids:
  - train: 70%
  - val: 15%
  - test: 15%

Why
If you train on nearly identical shapes and test on same shapes,
your predictor looks unrealistically good.
We want generalization across shapes.
