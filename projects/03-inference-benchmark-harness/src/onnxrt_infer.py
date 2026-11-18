from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.models as tvm

import onnxruntime as ort


@dataclass
class ONNXRTConfig:
    model_name: str = "resnet18"
    input_size: int = 224
    batch: int = 1
    opset: int = 17
    onnx_dir: str = "inference_bench/onnx"
    providers: Tuple[str, ...] = ("CPUExecutionProvider",)


class ONNXRTRunner:
    def __init__(self, cfg: ONNXRTConfig) -> None:
        self.cfg = cfg
        self.onnx_path = Path(cfg.onnx_dir) / f"{cfg.model_name}_{cfg.input_size}.onnx"
        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.onnx_path.exists():
            self._export_onnx()

        sess_opts = ort.SessionOptions()
        self.sess = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_opts,
            providers=list(cfg.providers),
        )
        self.input_name = self.sess.get_inputs()[0].name

    def _build_torch_model(self) -> torch.nn.Module:
        if self.cfg.model_name == "resnet18":
            m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
        elif self.cfg.model_name == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT)
        elif self.cfg.model_name == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {self.cfg.model_name}")
        m.eval()
        return m

    def _export_onnx(self) -> None:
        model = self._build_torch_model()

        dummy = torch.randn(self.cfg.batch, 3, self.cfg.input_size, self.cfg.input_size, dtype=torch.float32)
        torch.onnx.export(
            model,
            dummy,
            str(self.onnx_path),
            export_params=True,
            opset_version=self.cfg.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        )

    def forward(self, x_nchw_f32: np.ndarray) -> np.ndarray:
        if x_nchw_f32.dtype != np.float32:
            x_nchw_f32 = x_nchw_f32.astype(np.float32)
        out = self.sess.run(None, {self.input_name: x_nchw_f32})
        return out[0]
