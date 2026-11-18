"""Imagenette data: a 10-class ImageNet subset small enough for a free GPU.

Imagenette classes are real ImageNet classes, so ImageNet-pretrained models
can be scored on them with no fine-tuning: take the model's 1000 logits,
keep the 10 that belong to Imagenette classes, and take the argmax.
"""
from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# WordNet ID (folder name) -> ImageNet-1k class index
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888605": 701,  # parachute
}

# Standard ImageNet evaluation preprocessing
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def download_imagenette(root: str | Path = "data") -> Path:
    root = Path(root)
    target = root / "imagenette2-320"
    if target.exists():
        return target
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "imagenette2-320.tgz"
    print(f"Downloading {IMAGENETTE_URL} ...")
    urllib.request.urlretrieve(IMAGENETTE_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(root)
    archive.unlink()
    return target


def imagenet_class_indices(folder_classes: list[str]) -> torch.Tensor:
    """ImageNet indices ordered like ImageFolder's sorted class folders."""
    return torch.tensor([IMAGENETTE_TO_IMAGENET[c] for c in folder_classes])


def make_loaders(root: Path, batch_size: int, n_calib: int, seed: int = 0):
    """Returns (calibration loader from *train*, evaluation loader from *val*, class indices).

    Calibration images come from the training split and evaluation images from the
    validation split, so INT8 calibration never sees the images it is scored on.
    """
    train = datasets.ImageFolder(root / "train", EVAL_TRANSFORM)
    val = datasets.ImageFolder(root / "val", EVAL_TRANSFORM)
    g = torch.Generator().manual_seed(seed)
    calib_idx = torch.randperm(len(train), generator=g)[:n_calib].tolist()
    calib = DataLoader(Subset(train, calib_idx), batch_size=batch_size, num_workers=2)
    evaluation = DataLoader(val, batch_size=batch_size, num_workers=2)
    return calib, evaluation, imagenet_class_indices(val.classes)
