"""CNN-based patch tokenizer for ECG signals.

Replaces the linear projection used in the original ECG-JEPA with a shallow
1D CNN. Kernel sizes are chosen to cover physiologically relevant scales at
250 Hz sampling (kernel=15 → ~60 ms, comparable to QRS width).

The tokenizer is applied independently and identically to every (lead, time)
patch — cross-lead and cross-time context is the transformer's job.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import TokenizerConfig


class ECGTokenizer(nn.Module):
    def __init__(self, cfg: TokenizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size
        self.embed_dim = cfg.embed_dim

        self.conv1 = nn.Conv1d(
            1,
            cfg.conv1_channels,
            kernel_size=cfg.conv1_kernel,
            padding="same",
        )
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(
            cfg.conv1_channels,
            cfg.conv2_channels,
            kernel_size=cfg.conv2_kernel,
            padding="same",
        )
        self.act2 = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(cfg.conv2_channels, cfg.embed_dim)

    def patchify(self, signal: torch.Tensor) -> torch.Tensor:
        """Split a raw ECG signal into non-overlapping patches.

        Args:
            signal: (bs, C, T) raw multi-lead ECG.

        Returns:
            (bs, C, N, patch_size) where N = T // patch_size.
        """
        bs, c, t = signal.shape
        if t % self.patch_size != 0:
            raise ValueError(
                f"Signal length T={t} is not divisible by patch_size={self.patch_size}. "
                "Trim or pad the signal before tokenization."
            )
        n = t // self.patch_size
        return signal.reshape(bs, c, n, self.patch_size)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Tokenize patched ECG into D-dimensional token vectors.

        Args:
            patches: (bs, C, N, patch_size) output of `patchify` (or produced upstream).

        Returns:
            (bs, C, N, embed_dim)
        """
        bs, c, n, p = patches.shape
        if p != self.patch_size:
            raise ValueError(
                f"Expected patch_size={self.patch_size}, got {p}"
            )
        # Flatten (bs, C, N) into a single batch axis for the 1D CNN.
        x = patches.reshape(bs * c * n, 1, p)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool(x)            # (bs*C*N, conv2_channels, 1)
        x = x.squeeze(-1)           # (bs*C*N, conv2_channels)
        x = self.proj(x)            # (bs*C*N, embed_dim)
        return x.reshape(bs, c, n, self.embed_dim)
