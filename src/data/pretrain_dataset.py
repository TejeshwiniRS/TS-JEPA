"""Raw-signal pretraining dataset.

Loads the unified pretraining arrays produced by
``scripts.pretrain_data.build_pretrain_npy``:

  ``<data_dir>/X_pretrain_train.npy``   shape ``(N_train, 8, 2500)`` float32
  ``<data_dir>/X_pretrain_val.npy``     shape ``(N_val,   8, 2500)`` float32

Per the paper, signals are stored in raw microvolts (no normalization, no
filtering) and any record containing missing values has already been dropped
upstream by the data pipeline.

The arrays are memory-mapped (``mmap_mode='r'``) so that very large pretraining
corpora (CODE-15 alone is >200k samples) do not need to fit in RAM. Per-sample
copies are cheap (8 * 2500 * 4 bytes = 80 KB) and made on access.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PretrainECGDataset(Dataset):
    """Memory-mapped pretraining ECG dataset (no labels, no normalization)."""

    FILENAMES: dict[str, str] = {
        "train": "X_pretrain_train.npy",
        "val": "X_pretrain_val.npy",
    }

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        super().__init__()
        if split not in self.FILENAMES:
            raise ValueError(
                f"split must be one of {list(self.FILENAMES)}, got {split!r}"
            )
        path = Path(data_dir) / self.FILENAMES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrain data file not found: {path}\n"
                "Build it first via:\n"
                "  python -m scripts.pretrain_data.build_pretrain_npy --out_dir "
                f"{data_dir} ..."
            )
        self.path = path
        self.signals = np.load(str(path), mmap_mode="r")
        if self.signals.ndim != 3:
            raise ValueError(
                f"Expected 3D array (N, C, T), got shape {self.signals.shape} from {path}"
            )

    def __len__(self) -> int:
        return int(self.signals.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Materialize the row out of the mmap so the worker process can release
        # the underlying file handle quickly.
        sig = np.asarray(self.signals[idx], dtype=np.float32)
        return torch.from_numpy(sig)  # (C, T)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.signals.shape)  # type: ignore[return-value]


def get_pretrain_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    drop_last_train: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)`` for JEPA pretraining."""
    train_ds = PretrainECGDataset(data_dir=data_dir, split="train")
    val_ds = PretrainECGDataset(data_dir=data_dir, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
