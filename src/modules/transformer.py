"""Pre-LN transformer block shared by the encoder and the predictor."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class DropPath(nn.Module):
    """Stochastic depth per sample. Drops the residual branch with probability `p`."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        mask.div_(keep_prob)
        return x * mask


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_flash: bool = True,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_flash=use_flash,
            qkv_bias=qkv_bias,
            dropout=dropout,
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.norm1(x), return_attn=return_attn)
        if return_attn:
            attn_out, attn_weights = attn_out
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn_weights
        return x
