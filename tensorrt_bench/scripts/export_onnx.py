import argparse
import torch
import torchvision.models as models

MODEL_MAP = {
    "resnet18": models.resnet18,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "efficientnet_b0": models.efficientnet_b0,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=MODEL_MAP.keys())
    ap.add_argument("--out", type=str, default="results/model.onnx")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    m = MODEL_MAP[args.model](weights="DEFAULT")
    m.eval()

    x = torch.randn(1, 3, args.input_size, args.input_size)

    torch.onnx.export(
        m,
        x,
        args.out,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # fixed-shape baseline
    )

    print(f"Exported ONNX: {args.out}")

if __name__ == "__main__":
    main()
