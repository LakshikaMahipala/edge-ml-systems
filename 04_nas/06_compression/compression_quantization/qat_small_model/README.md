QAT Small Model

What this demonstrates
- FP32 training baseline
- QAT training (fake quant + observers)
- conversion to an INT8 quantized model
- simple metrics + inference timing

Run later
python src/train_fp32.py --epochs 10 --out results/fp32_metrics.json
python src/train_qat.py  --epochs 10 --out results/qat_metrics.json
