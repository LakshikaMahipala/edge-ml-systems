Mini-project 8 
Predictor-based schedule selection + FPGA sweep as second target.

Contents
- docs/: problem, method, features, models, selection algorithm
- src/: merge helper, ranker scaffold, regret@k simulator
- results/: template_report.md

Run later (example)
python src/merge_datasets_fpga_and_layer.py
python src/rank_candidates_with_predictor.py --candidates_csv results/merged_dataset.csv
python src/selection_simulator.py --ranked_csv results/ranked.csv --label_col y_fpga_est_total_ms
