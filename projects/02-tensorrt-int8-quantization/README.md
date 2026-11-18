# TensorRT FP32 / FP16 / INT8: Accuracy vs Speed on an Affordable GPU

> How much faster does INT8 TensorRT make a CNN on an NVIDIA T4, and how much accuracy does it cost? Measured end to end, from PyTorch model to calibrated INT8 engine.

**Status:** 🟡 Pipeline complete · T4 results pending (run the [Colab notebook](notebooks/run_on_colab.ipynb), ~15 min)

## Question
Post-training INT8 quantization promises big speed-ups, but the gain depends on the model and the GPU, and the accuracy cost depends on calibration. This project measures both on one budget GPU (T4, Turing, INT8 tensor cores) with a real labelled dataset instead of random inputs.

## Method
```
torchvision model (ImageNet weights)
   └─► ONNX (opset 17, dynamic batch)
         ├─► TensorRT FP32
         ├─► TensorRT FP16
         └─► TensorRT INT8  ◄── entropy calibration on 512 Imagenette *train* images
               │
               ▼
   Top-1 on Imagenette *val* (3,925 images)   +   GPU latency p50/p99 (batch 1)   +   throughput (batch 32)
```
- **Dataset:** [Imagenette](https://github.com/fastai/imagenette), 10 real ImageNet classes. ImageNet-pretrained models are scored with no fine-tuning by taking the argmax over those 10 classes' logits.
- **No data leakage:** calibration images come from the train split, evaluation images from the val split.
- **Baseline:** the same model in PyTorch FP32 eager mode.
- **Timing:** CUDA events, 20 warm-up and 200 timed inferences, GPU time only.
- **Models:** ResNet-18 (compute-heavy) and MobileNetV3-Large (depthwise convolutions, which gain less from INT8).

## Results (NVIDIA T4)
_Pending: commit `results/t4/results.csv` and `results/t4/accuracy_vs_latency.png` after running the notebook._

| Model | Backend | Precision | Top-1 | p50 latency, batch 1 (ms) | Throughput, batch 32 (img/s) | Engine (MB) |
|---|---|---|---|---|---|---|
| ResNet-18 | PyTorch | FP32 | – | – | – | – |
| ResNet-18 | TensorRT | FP32 | – | – | – | – |
| ResNet-18 | TensorRT | FP16 | – | – | – | – |
| ResNet-18 | TensorRT | INT8 | – | – | – | – |

## Layout
```
├── src/
│   ├── data.py           # Imagenette download, ImageNet class mapping, loaders
│   ├── trt_engine.py     # engine builder, INT8 entropy calibrator, runner (TensorRT 10 API)
│   ├── run_benchmark.py  # full experiment → results.csv
│   └── plot_results.py   # accuracy-vs-latency and throughput plots
├── notebooks/run_on_colab.ipynb
└── docs/                 # TensorRT playbook, INT8 PTQ notes, DLA notes (Jetson)
```

## Reproduce
```bash
pip install -r requirements.txt          # needs an NVIDIA GPU + driver
python src/run_benchmark.py --models resnet18 mobilenet_v3_large --out results/t4
python src/plot_results.py results/t4/results.csv results/t4/accuracy_vs_latency.png
```

## Limitations and next steps
- [ ] Commit T4 results
- [ ] Compare calibration set sizes (64 / 256 / 1024 images) and calibrators (entropy vs min-max vs percentile)
- [ ] Per-layer sensitivity: keep the most error-prone layers in FP16
- [ ] Repeat on a Jetson Orin Nano (GPU + DLA)
