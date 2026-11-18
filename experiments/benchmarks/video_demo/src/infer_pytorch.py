import time
import numpy as np
import torch
import torchvision.models as models

class PyTorchVideoInfer:
    def __init__(self, model_name: str = "resnet18", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load(model_name).to(self.device).eval()

    def _load(self, name: str):
        if name == "resnet18":
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if name == "mobilenet_v3_small":
            return models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        if name == "efficientnet_b0":
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        raise ValueError(f"Unknown model: {name}")

    @torch.inference_mode()
    def infer(self, x_nchw: np.ndarray):
        x = torch.from_numpy(x_nchw).to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = self.model(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        probs = torch.softmax(logits, dim=1)
        top1 = int(torch.argmax(probs, dim=1).item())
        conf = float(torch.max(probs, dim=1).values.item())
        latency_ms = (t1 - t0) * 1e3
        return top1, conf, latency_ms
