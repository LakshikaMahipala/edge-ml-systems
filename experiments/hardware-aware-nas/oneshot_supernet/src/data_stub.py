from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticCIFAR(Dataset):
    def __init__(self, n: int = 5000, num_classes: int = 10, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.X = torch.randn(n, 3, 32, 32, generator=g)
        self.y = torch.randint(0, num_classes, (n,), generator=g)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loaders(batch: int = 64, seed: int = 0):
    train = SyntheticCIFAR(n=4096, seed=seed)
    val   = SyntheticCIFAR(n=1024, seed=seed + 1)
    return (
        DataLoader(train, batch_size=batch, shuffle=True),
        DataLoader(val, batch_size=batch, shuffle=True),
    )
