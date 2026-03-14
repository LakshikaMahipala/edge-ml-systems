Latency predictor modeling plan 

Target
Predict y_fpga_est_total_ms from layer configs.

Models
A) Tree-based baseline (scikit-learn HistGradientBoostingRegressor)
- strong for tabular features
- handles non-linearities without scaling requirements
- good baseline to beat

B) MLP regressor (PyTorch)
- learns continuous mappings in log-space well
- requires scaling + careful training

Key choice: predict log latency
We train on:
  y = log(1 + latency_ms)
This makes learning stable across wide ranges.

Evaluation
- MAE (ms)
- RMSE (ms)
- MAPE (%)
- Spearman rank correlation (how well model preserves ordering)
- Report by op_type buckets (FC vs DWConv)

No leakage rule
Use group split by config_id (same as dataset protocol).
