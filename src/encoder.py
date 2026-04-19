"""ECG-JEPA encoder.

Used as both the context (student) encoder and the target (teacher) encoder;
the training loop owns two instances and updates the teacher via EMA.

Key behavior:
- Input is the patched ECG (bs, C, N, patch_size). Patching logic lives on the
  tokenizer.
- Positional embeddings are added BEFORE masking so each token carries its
  true (lead, time) position regardless of whether it is later dropped.
- Masking is synchronized across leads. The caller passes `visible_indices`:
  a 1D LongTensor of temporal indices ∈ [0, N) that are kept. All leads use
  the same temporal indices.
- No CroPA — full self-attention over visible tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs import EncoderConfig
from .modules import TransformerBlock
from .pos_encoding import get_2d_sincos_pos_embed
from .tokenizer import ECGTokenizer


class ECGEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, tokenizer: ECGTokenizer) -> None:
        super().__init__()
        if tokenizer.embed_dim != cfg.embed_dim:
            raise ValueError(
                f"tokenizer.embed_dim ({tokenizer.embed_dim}) must match "
                f"encoder.embed_dim ({cfg.embed_dim})"
            )
        if tokenizer.patch_size != cfg.patch_size:
            raise ValueError(
                f"tokenizer.patch_size ({tokenizer.patch_size}) must match "
                f"encoder.patch_size ({cfg.patch_size})"
            )

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.num_leads = cfg.num_leads
        self.num_patches = cfg.num_patches
        self.embed_dim = cfg.embed_dim

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=cfg.embed_dim,
            num_leads=cfg.num_leads,
            num_patches=cfg.num_patches,
        )  # (C*N, D)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

        # Linear drop-path schedule across layers.
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

    def forward(
        self,
        patches: torch.Tensor,
        visible_indices: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Encode patched ECG into contextualized token representations.

        Args:
            patches: (bs, C, N, patch_size) patched ECG.
            visible_indices: Optional LongTensor of shape (Q,) with temporal
                indices ∈ [0, N) that are visible to this encoder. If None,
                all N temporal positions are used (target-encoder / inference
                path). Masking is synchronized across leads, so the same Q
                indices are applied to every lead.
            return_attn: If True, additionally return per-block attention maps.
                Only supported when the module was built with use_flash=False.

        Returns:
            (bs, C*num_visible, D) contextualized representations.
            num_visible = Q if `visible_indices` is provided, else N.
            If return_attn=True, also returns a list of (bs, H, S, S) tensors,
            one per block.
        """
        bs, c, n, _ = patches.shape
        if c != self.num_leads:
            raise ValueError(
                f"Expected C={self.num_leads} leads, got {c}"
            )
        if n != self.num_patches:
            raise ValueError(
                f"Expected N={self.num_patches} patches, got {n}"
            )

        tokens = self.tokenizer(patches)                     # (bs, C, N, D)
        tokens = tokens.reshape(bs, c * n, self.embed_dim)   # (bs, C*N, D)
        tokens = tokens + self.pos_embed.unsqueeze(0)        # broadcast over batch

        if visible_indices is not None:
            tokens = tokens.reshape(bs, c, n, self.embed_dim)
            tokens = tokens.index_select(dim=2, index=visible_indices)  # (bs, C, Q, D)
            tokens = tokens.reshape(bs, c * tokens.shape[2], self.embed_dim)

        attn_maps: list[torch.Tensor] = []
        for block in self.blocks:
            out = block(tokens, return_attn=return_attn)
            if return_attn:
                tokens, attn_w = out
                attn_maps.append(attn_w)
            else:
                tokens = out

        tokens = self.norm(tokens)

        if return_attn:
            return tokens, attn_maps
        return tokens

    def forward_all(
        self,
        patches: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Convenience wrapper: encode all C*N tokens (no masking).

        Used by the target encoder and at inference time.
        """
        return self.forward(patches, visible_indices=None, return_attn=return_attn)
