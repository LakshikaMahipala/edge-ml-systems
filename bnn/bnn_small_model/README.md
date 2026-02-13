BNN Small Model 

What this demo shows
- FP32 baseline MLP
- BNN MLP using STE sign for binarization
- BatchNorm + bounded activations to stabilize training
- Compare accuracy + inference timing (proxy)

Run later
python src/train_fp32.py --epochs 10
python src/train_bnn.py  --epochs 15
