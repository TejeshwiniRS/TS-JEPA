"""Masking strategies for JEPA pretraining.

All strategies return a (visible_indices, masked_indices) pair of sorted
1D LongTensors on the specified device. Masking is synchronized across leads.
"""

from __future__ import annotations

import torch


def random_mask(
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly random patch masking (I-JEPA style).

    Args:
        num_patches: Total number of temporal patches N.
        mask_ratio: Fraction of patches to mask.
        device: Target device.

    Returns:
        (visible_indices, masked_indices) — sorted LongTensors on *device*.
    """
    num_mask = int(round(num_patches * mask_ratio))
    num_mask = max(1, min(num_mask, num_patches - 1))
    perm = torch.randperm(num_patches, device=device)
    masked = perm[:num_mask].sort().values
    visible = perm[num_mask:].sort().values
    return visible, masked


def block_mask(
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal block masking — contiguous future block is masked (causal setup).

    The visible patches are the first (1 - mask_ratio) * N positions;
    the masked patches are the trailing block.
    """
    num_mask = int(round(num_patches * mask_ratio))
    num_mask = max(1, min(num_mask, num_patches - 1))
    num_visible = num_patches - num_mask
    visible = torch.arange(0, num_visible, device=device)
    masked = torch.arange(num_visible, num_patches, device=device)
    return visible, masked


def get_mask_fn(strategy: str):
    """Return the masking function for the given strategy name."""
    _registry = {
        "random": random_mask,
        "block": block_mask,
    }
    if strategy not in _registry:
        raise ValueError(
            f"Unknown mask strategy '{strategy}'. Choose from {list(_registry.keys())}"
        )
    return _registry[strategy]
