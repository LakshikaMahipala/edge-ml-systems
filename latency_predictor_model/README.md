Latency Predictor Model 

Purpose
Train latency predictors on the dataset built in latency_predictor_dataset.

Models
- train_sklearn_baseline.py (tabular baseline)
- train_mlp_torch.py (neural baseline)

Evaluation
- MAE/RMSE/MAPE
- Spearman rank correlation (critical for ranking-based search)

Run later
python src/train_sklearn_baseline.py --dataset_csv ../latency_predictor_dataset/data/dataset.csv
python src/train_mlp_torch.py --dataset_csv ../latency_predictor_dataset/data/dataset.csv
