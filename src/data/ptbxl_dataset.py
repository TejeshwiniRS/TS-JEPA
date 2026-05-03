"""PTB-XL dataset (downstream evaluation only).

Pretraining now uses ``PretrainECGDataset`` over the unified
Chapman + Ningbo + CODE-15 npy arrays. PTB-XL is reserved for downstream
linear-probe / fine-tuning experiments.

Expected files in `data_dir`:
    X_ecg_train.npy   (17441, 12, 1000)  float32
    X_ecg_val.npy     (2193,  12, 1000)  float32
    X_ecg_test.npy    (2203,  12, 1000)  float32
    y_ecg_train.npy   (17441, 5)         float32  - multi-label, 5 PTB-XL superclasses
    y_ecg_val.npy     (2193,  5)         float32
    y_ecg_test.npy    (2203,  5)         float32
    norm_ecg_mean.npy (1, 12, 1)         float32  - per-lead normalization mean
    norm_ecg_std.npy  (1, 12, 1)         float32  - per-lead normalization std
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PTBXLDataset(Dataset):
    """PTB-XL from pre-saved npy files.

    For self-supervised pretraining, labels are ignored — only the signal is
    returned.  When `return_labels=True` the dataset also returns the 5-class
    multi-label target (useful for downstream linear probing / fine-tuning).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        normalize: bool = False,
        return_labels: bool = False,
    ) -> None:
        super().__init__()
        data_dir = Path(data_dir)

        split_map = {"train": "train", "val": "val", "test": "test"}
        if split not in split_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        x_path = data_dir / f"X_ecg_{split}.npy"
        y_path = data_dir / f"y_ecg_{split}.npy"
        if not x_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {x_path}. "
                "Place the PTB-XL npy files in the data directory."
            )

        self.signals = np.load(str(x_path)).astype(np.float32)  # (N, 12, T)
        self.labels = np.load(str(y_path)).astype(np.float32) if y_path.exists() else None
        self.return_labels = return_labels

        if normalize:
            mean_path = data_dir / "norm_ecg_mean.npy"
            std_path = data_dir / "norm_ecg_std.npy"
            if mean_path.exists() and std_path.exists():
                mean = np.load(str(mean_path)).astype(np.float32)  # (1, 12, 1)
                std = np.load(str(std_path)).astype(np.float32)    # (1, 12, 1)
                self.signals = (self.signals - mean) / np.clip(std, 1e-8, None)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int):
        sig = torch.from_numpy(self.signals[idx])  # (12, T)
        if self.return_labels and self.labels is not None:
            lbl = torch.from_numpy(self.labels[idx])  # (5,)
            return sig, lbl
        return sig


def get_ptbxl_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    return_labels: bool = True,
    normalize: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train, val, test)`` PTB-XL loaders for downstream evaluation."""
    train_ds = PTBXLDataset(
        data_dir=data_dir, split="train",
        normalize=normalize, return_labels=return_labels,
    )
    val_ds = PTBXLDataset(
        data_dir=data_dir, split="val",
        normalize=normalize, return_labels=return_labels,
    )
    test_ds = PTBXLDataset(
        data_dir=data_dir, split="test",
        normalize=normalize, return_labels=return_labels,
    )
    common = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, test_loader
