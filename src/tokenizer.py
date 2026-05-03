"""Per-patch tokenizer for ECG signals.

Two implementations are exposed via `TokenizerConfig.kind`:

  - ``"linear"`` (default, paper-faithful):
        a single ``nn.Linear(patch_size -> embed_dim)``. The ECG-JEPA paper
        describes patches as being projected by "a linear layer" before
        positional embeddings are added.

  - ``"ffn"`` (opt-in for ablations):
        ``Linear -> GELU -> Linear`` with `ffn_hidden_dim` in the middle.

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
        self.kind = cfg.kind

        if cfg.kind == "linear":
            self.proj: nn.Module = nn.Linear(cfg.patch_size, cfg.embed_dim)
        elif cfg.kind == "ffn":
            self.proj = nn.Sequential(
                nn.Linear(cfg.patch_size, cfg.ffn_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.ffn_hidden_dim, cfg.embed_dim),
            )
        else:
            raise ValueError(
                f"TokenizerConfig.kind must be 'linear' or 'ffn', got {cfg.kind!r}"
            )

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
        # Flatten (bs, C, N) into a single batch axis for per-patch projection.
        x = patches.reshape(bs * c * n, p)
        x = self.proj(x)            # (bs*C*N, embed_dim)
        return x.reshape(bs, c, n, self.embed_dim)
