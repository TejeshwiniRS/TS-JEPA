"""Masking strategies for JEPA pretraining.

All strategies return a `(visible_indices, masked_indices)` pair of sorted
1D LongTensors on the requested device. Masking is **synchronized across
leads**: the same temporal indices are masked in every channel of every
sample in the batch.

Three strategies are exposed via :func:`get_mask_fn`:

  - ``"random"``      — I-JEPA style uniform random masking.
  - ``"block"``       — single trailing block (kept for back-compat).
  - ``"multi_block"`` — paper-faithful overlapping multi-block masking
                        (ECG-JEPA Section 3.1). Multiple blocks of variable
                        size are sampled and may overlap; the union of their
                        positions is the masked set.

Different strategies have different `mask_ratio` semantics:

  - ``random`` and ``block`` interpret `mask_ratio` directly as the fraction
    of patches to mask.
  - ``multi_block`` interprets it as the per-block fraction and additionally
    needs `freq` (number of blocks); the realized total mask ratio is
    typically ~freq * ratio (less, due to overlap).
"""

from __future__ import annotations

from typing import Callable

import torch


def random_mask(
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str = "cpu",
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly random patch masking (I-JEPA style)."""
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
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single contiguous trailing block (kept for back-compat with old runs)."""
    num_mask = int(round(num_patches * mask_ratio))
    num_mask = max(1, min(num_mask, num_patches - 1))
    num_visible = num_patches - num_mask
    visible = torch.arange(0, num_visible, device=device)
    masked = torch.arange(num_visible, num_patches, device=device)
    return visible, masked


def multi_block_mask(
    num_patches: int,
    mask_ratio: float,
    device: torch.device | str = "cpu",
    *,
    freq: int = 4,
    block_size_min: int | None = None,
    block_size_max: int | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper-faithful overlapping multi-block masking.

    Samples ``freq`` blocks. Each block has length drawn uniformly from
    ``[block_size_min, block_size_max]`` (inclusive); if those bounds are not
    supplied, they default to ``round(mask_ratio * num_patches)`` so the
    caller can drive everything from the ``(mask_ratio_min, mask_ratio_max)``
    schedule. Block start positions are uniform in ``[0, N - block_size]``.
    Blocks may overlap — the masked set is the union of their indices.

    The visible set is the complement. We always guarantee at least one
    masked and one visible patch.
    """
    if freq < 1:
        raise ValueError(f"multi_block_mask requires freq >= 1, got {freq}")
    if block_size_min is None or block_size_max is None:
        size = max(1, int(round(mask_ratio * num_patches)))
        block_size_min = block_size_min or size
        block_size_max = block_size_max or size
    if block_size_min > block_size_max:
        raise ValueError(
            f"block_size_min ({block_size_min}) > block_size_max ({block_size_max})"
        )
    block_size_min = max(1, min(block_size_min, num_patches - 1))
    block_size_max = max(block_size_min, min(block_size_max, num_patches - 1))

    masked_set: set[int] = set()
    for _ in range(freq):
        size = int(torch.randint(block_size_min, block_size_max + 1, (1,)).item())
        # Uniform start position; allow overlap with prior blocks.
        max_start = num_patches - size
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        masked_set.update(range(start, start + size))

    # Guarantee at least one visible patch.
    if len(masked_set) >= num_patches:
        # Drop a random index from the masked set so something stays visible.
        drop = int(torch.randint(0, num_patches, (1,)).item())
        masked_set.discard(drop)
    # Guarantee at least one masked patch (extremely unlikely to fail).
    if not masked_set:
        masked_set.add(int(torch.randint(0, num_patches, (1,)).item()))

    masked = torch.tensor(sorted(masked_set), dtype=torch.long, device=device)
    all_idx = torch.arange(num_patches, device=device)
    visible_set = set(range(num_patches)) - masked_set
    visible = torch.tensor(sorted(visible_set), dtype=torch.long, device=device)
    del all_idx
    return visible, masked


_MaskFn = Callable[..., tuple[torch.Tensor, torch.Tensor]]


def get_mask_fn(strategy: str) -> _MaskFn:
    """Return the masking function for the given strategy name."""
    _registry: dict[str, _MaskFn] = {
        "random": random_mask,
        "block": block_mask,
        "multi_block": multi_block_mask,
    }
    if strategy not in _registry:
        raise ValueError(
            f"Unknown mask strategy '{strategy}'. Choose from {list(_registry.keys())}"
        )
    return _registry[strategy]
