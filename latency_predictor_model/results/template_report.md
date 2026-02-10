# Latency Predictor Report 

## Dataset
- rows:
- ops:
- target:

## Split
- group split by config_id
- train/val/test counts:

## Models
### sklearn baseline (HistGradientBoostingRegressor)
- val: MAE / RMSE / MAPE / Spearman
- test: MAE / RMSE / MAPE / Spearman

### PyTorch MLP
- val: ...
- test: ...

## Key interpretation
- If Spearman is high: predictor is useful for ranking in autotuning/NAS.
- If MAE is low: predictor is useful for absolute latency estimates.
- If UART dominates, most signal will come from bytes_total, not macs.
