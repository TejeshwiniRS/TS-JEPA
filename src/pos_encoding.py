"""Sinusoidal positional embeddings for ECG-JEPA.

- 1D variant: used by the predictor (temporal positions only, one sequence per lead).
- 2D variant: used by the encoder (joint lead + temporal position over a C×N grid).

Both are non-learnable and deterministic; the caller registers them as buffers.
"""

from __future__ import annotations

import numpy as np
import torch


def get_1d_sincos_pos_embed(embed_dim: int, num_positions: int) -> torch.Tensor:
    """Standard 1D sinusoidal positional embedding.

    Args:
        embed_dim: Embedding dimension. Must be even.
        num_positions: Number of positions (e.g. N temporal patches).

    Returns:
        Tensor of shape (num_positions, embed_dim), dtype float32.
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    positions = np.arange(num_positions, dtype=np.float64)
    return torch.from_numpy(_sincos_from_positions(embed_dim, positions)).float()


def get_2d_sincos_pos_embed(
    embed_dim: int,
    num_leads: int,
    num_patches: int,
) -> torch.Tensor:
    """2D sinusoidal positional embedding over a (lead, temporal) grid.

    The embed_dim is split in half: the first half encodes the lead index,
    the second half encodes the temporal index. Tokens are laid out in
    row-major order (lead is the slow index, time is the fast index), so
    a flattened tensor of shape (num_leads * num_patches, embed_dim) matches
    the order produced by `patches.reshape(bs, C * N, ...)`.

    Args:
        embed_dim: Embedding dimension. Must be divisible by 4 (split in half
            for lead vs. time, and each half must itself be even).
        num_leads: Number of ECG leads (C).
        num_patches: Number of temporal patches (N).

    Returns:
        Tensor of shape (num_leads * num_patches, embed_dim), dtype float32.
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"embed_dim must be divisible by 4 for 2D sincos, got {embed_dim}"
        )
    half_dim = embed_dim // 2

    lead_idx = np.arange(num_leads, dtype=np.float64)
    time_idx = np.arange(num_patches, dtype=np.float64)

    lead_embed = _sincos_from_positions(half_dim, lead_idx)  # (C, half)
    time_embed = _sincos_from_positions(half_dim, time_idx)  # (N, half)

    # Broadcast to (C, N, half) for each half, then concat on last dim.
    lead_grid = np.broadcast_to(lead_embed[:, None, :], (num_leads, num_patches, half_dim))
    time_grid = np.broadcast_to(time_embed[None, :, :], (num_leads, num_patches, half_dim))
    grid = np.concatenate([lead_grid, time_grid], axis=-1)  # (C, N, embed_dim)
    grid = grid.reshape(num_leads * num_patches, embed_dim)
    return torch.from_numpy(grid).float()


def _sincos_from_positions(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    """Core sinusoidal formula shared by the 1D and 2D helpers.

    Args:
        embed_dim: Output dimension. Must be even.
        positions: 1D array of positions, shape (P,).

    Returns:
        Array of shape (P, embed_dim).
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"embed_dim must be even, got {embed_dim}")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0**omega)  # (embed_dim/2,)

    out = np.einsum("p,d->pd", positions, omega)  # (P, embed_dim/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (P, embed_dim)
