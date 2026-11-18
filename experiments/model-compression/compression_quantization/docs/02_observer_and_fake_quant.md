# Observers and FakeQuant

Observer:
- watches activations during calibration/training
- estimates min/max (or histogram-based) to set quantization params

FakeQuant:
- during training, inserts "quantize -> dequantize" in forward
- gradient uses STE so the model can adapt to quantization noise
