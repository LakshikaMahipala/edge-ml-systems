# Validation protocol

We care about:
1) ranking quality (NAS needs ordering)
   - Spearman correlation
   - Kendall tau
2) absolute error (deployment budgeting)
   - MAPE
   - RMSE

Always validate on:
- architectures not seen in training (holdout graphs)
- ideally new shapes (generalization test)
