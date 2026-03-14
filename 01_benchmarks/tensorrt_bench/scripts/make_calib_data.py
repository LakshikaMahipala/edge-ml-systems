import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="results/calib/calib_fp32.npy")
    ap.add_argument("--n", type=int, default=256, help="number of calibration samples")
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = FakeData(
        size=args.n,
        image_size=(3, args.input_size, args.input_size),
        num_classes=1000,
        transform=tfm
    )

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    xs = []
    for x, _y in dl:
        xs.append(x.numpy().astype(np.float32))

    X = np.concatenate(xs, axis=0)  # (N,3,H,W)
    np.save(out, X)
    print(f"Saved calibration tensor: {out} shape={X.shape} dtype={X.dtype}")
    print("Note: FakeData is a placeholder. Replace with real images later for meaningful calibration.")


if __name__ == "__main__":
    main()
