"""Dataset wrapper for the preprocessed BCIC-IV 2a splits."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BCICDataset(Dataset):
    """Loads a single .npz split (train/val/test) into memory.

    Shapes: X (N, 22, 1000) float32, y (N,) int64. Total dataset is small
    (<2k trials) so we keep everything in RAM.
    """

    def __init__(self, npz_path: str | Path) -> None:
        d = np.load(npz_path)
        self.X = torch.from_numpy(d["X"]).float()
        self.y = torch.from_numpy(d["y"]).long()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]
