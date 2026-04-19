"""Multi-head self-attention with optional Flash Attention backend.

Two backends are supported, selected by the `use_flash` flag at construction:

- `use_flash=True`: uses `flash_attn.flash_attn_func` from the flash-attn library.
  Fast, memory-efficient, but never materializes the attention matrix, so
  attention weights are not available for inspection.

- `use_flash=False`: manual scaled dot-product attention. Slower, but supports
  `return_attn=True` which returns the full attention-weight tensor so that
  the caller can analyze attention maps after training.

The flag is fixed at construction time — swapping requires rebuilding the module.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_flash: bool = True,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_flash = use_flash
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self._flash_fn = None
        if use_flash:
            try:
                from flash_attn import flash_attn_func  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "use_flash=True requires the `flash-attn` package. "
                    "Install it or construct the module with use_flash=False."
                ) from e
            self._flash_fn = flash_attn_func

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (bs, seq_len, embed_dim)
            return_attn: If True (standard backend only), also return attention weights
                of shape (bs, num_heads, seq_len, seq_len).

        Returns:
            (bs, seq_len, embed_dim), optionally plus attention weights.
        """
        bs, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(bs, seq_len, 3, self.num_heads, self.head_dim)

        if self.use_flash:
            if return_attn:
                raise ValueError(
                    "return_attn=True is not supported with use_flash=True. "
                    "Construct the module with use_flash=False to access attention maps."
                )
            # flash_attn expects (bs, seq_len, num_heads, head_dim) for q, k, v
            q, k, v = qkv.unbind(dim=2)
            drop_p = self.dropout if self.training else 0.0
            out = self._flash_fn(q, k, v, dropout_p=drop_p, causal=False)
            out = out.reshape(bs, seq_len, self.embed_dim)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

        # Standard scaled-dot-product attention.
        # Rearrange to (bs, num_heads, seq_len, head_dim).
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (bs, H, S, S)
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn  # save pre-dropout for returning
        attn = self.attn_drop(attn)

        out = attn @ v  # (bs, H, S, head_dim)
        out = out.transpose(1, 2).reshape(bs, seq_len, self.embed_dim)
        out = self.proj(out)
        out = self.proj_drop(out)

        if return_attn:
            return out, attn_weights
        return out
