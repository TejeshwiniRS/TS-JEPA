"""ECG-JEPA predictor.

Takes the context encoder's output (visible tokens across all leads), projects
it into a smaller predictor dimension, appends learnable mask tokens at the
masked temporal positions, and runs a transformer **per lead independently**.
Its outputs at the masked positions are compared against the target encoder's
outputs at the same positions during training.

Key shape notes:
- The encoder output arrives as (bs, C*Q, encoder_embed_dim). Internally we
  reshape to (bs*C, Q, embed_dim) and process each lead as its own sequence,
  so the predictor's self-attention is **intra-lead only**. The cross-lead
  context is already baked into each visible token by the encoder.
- After transformer blocks run over Q+M tokens per lead, we slice out only
  the last M positions (the mask-token predictions) and return them projected
  back to encoder_embed_dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import PredictorConfig
from .modules import TransformerBlock
from .pos_encoding import get_1d_sincos_pos_embed


class ECGPredictor(nn.Module):
    def __init__(self, cfg: PredictorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_leads = cfg.num_leads
        self.num_patches = cfg.num_patches
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.embed_dim = cfg.embed_dim

        self.input_proj = nn.Linear(cfg.encoder_embed_dim, cfg.embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        pos_embed = get_1d_sincos_pos_embed(
            embed_dim=cfg.embed_dim,
            num_positions=cfg.num_patches,
        )  # (N, D)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        dpr = [x.item() for x in torch.linspace(0.0, cfg.drop_path, cfg.depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=cfg.embed_dim,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    drop_path=dpr[i],
                    use_flash=cfg.use_flash,
                    qkv_bias=cfg.qkv_bias,
                )
                for i in range(cfg.depth)
            ]
        )
        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.output_proj = nn.Linear(cfg.embed_dim, cfg.encoder_embed_dim)

    def forward(
        self,
        context_tokens: torch.Tensor,
        visible_indices: torch.Tensor,
        masked_indices: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Predict target representations at masked temporal positions.

        Args:
            context_tokens: (bs, C*Q, encoder_embed_dim) — context encoder output.
            visible_indices: LongTensor (Q,) — temporal indices of visible patches.
            masked_indices: LongTensor (M,) — temporal indices of masked patches.
                Together, visible_indices ∪ masked_indices = [0, N).
            return_attn: If True, return per-block attention maps. Only supported
                with use_flash=False.

        Returns:
            (bs, C, M, encoder_embed_dim) predicted representations at masked
            positions. The training loop compares these against the target
            encoder's outputs at the same (lead, masked-temporal) positions.
            If return_attn=True, also returns a list of (bs*C, H, S, S) attention
            tensors, one per block.
        """
        bs, cq, d_in = context_tokens.shape
        if d_in != self.encoder_embed_dim:
            raise ValueError(
                f"context_tokens last dim ({d_in}) must match "
                f"encoder_embed_dim ({self.encoder_embed_dim})"
            )
        c = self.num_leads
        q = cq // c
        if q * c != cq:
            raise ValueError(
                f"context_tokens length ({cq}) is not divisible by num_leads ({c})"
            )
        if visible_indices.numel() != q:
            raise ValueError(
                f"visible_indices has {visible_indices.numel()} entries but "
                f"context_tokens implies Q={q}"
            )
        m = masked_indices.numel()

        # Project to predictor dim: (bs, C*Q, D_pred).
        x = self.input_proj(context_tokens)

        # Split per-lead: (bs, C, Q, D) → (bs*C, Q, D).
        x = x.reshape(bs, c, q, self.embed_dim)
        x = x.reshape(bs * c, q, self.embed_dim)

        # Add visible temporal positional embeddings (same for every lead).
        vis_pos = self.pos_embed.index_select(dim=0, index=visible_indices)  # (Q, D)
        x = x + vis_pos.unsqueeze(0)

        # Build mask-token stream: (bs*C, M, D) with masked positional embeddings added.
        mask_tokens = self.mask_token.expand(bs * c, m, self.embed_dim)
        mask_pos = self.pos_embed.index_select(dim=0, index=masked_indices)  # (M, D)
        mask_tokens = mask_tokens + mask_pos.unsqueeze(0)

        # Concatenate: visible tokens first, mask tokens second.
        x = torch.cat([x, mask_tokens], dim=1)  # (bs*C, Q+M, D)

        attn_maps: list[torch.Tensor] = []
        for block in self.blocks:
            out = block(x, return_attn=return_attn)
            if return_attn:
                x, attn_w = out
                attn_maps.append(attn_w)
            else:
                x = out

        x = self.norm(x)
        x = self.output_proj(x)  # (bs*C, Q+M, encoder_embed_dim)

        # Keep only the mask-token predictions (last M positions).
        preds = x[:, q:, :]                                   # (bs*C, M, D_enc)
        preds = preds.reshape(bs, c, m, self.encoder_embed_dim)

        if return_attn:
            return preds, attn_maps
        return preds
