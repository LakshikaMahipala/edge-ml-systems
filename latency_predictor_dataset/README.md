Latency Predictor Dataset 

What it is
A dataset pipeline for learning latency predictors:
X = layer config features
y = latency labels (estimated now, measured later)

Current labels
- y_fpga_est_* derived from UART + cycle estimator

Files
- docs/: schema, feature engineering, labeling strategy, split protocol
- src/: generator, labeler, dataset builder
- data/: schema.json + dataset_template.csv

Run later
python src/generate_layer_configs.py --n 500 --seed 0 --out data/configs.jsonl
python src/label_with_fpga_estimator.py --in_jsonl data/configs.jsonl --out_jsonl data/labeled.jsonl
python src/make_dataset.py --in_jsonl data/labeled.jsonl --out_csv data/dataset.csv
