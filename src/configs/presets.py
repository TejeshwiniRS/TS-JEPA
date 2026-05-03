from __future__ import annotations

from .tokenizer_config import TokenizerConfig
from .encoder_config import EncoderConfig
from .predictor_config import PredictorConfig


# ECG-JEPA reference setup (Kim 2026, Section 4 + Section 4.3):
#   8 leads (I, II, V1-V6), 10 s @ 250 Hz -> T = 2500, patch_size = 50, N = 50.
# Final encoder: 12 layers, 16 heads, D = 768.
# Final predictor: 6 layers, 12 heads, D = 384 (smaller transformer).
NUM_LEADS = 8
NUM_PATCHES = 50  # T=2500, patch_size=50 -> 50 temporal patches.


def dev_preset() -> tuple[TokenizerConfig, EncoderConfig, PredictorConfig]:
    """Smaller architecture for ablations / CPU smoke tests.

    Same input geometry as `final_preset` (8 leads, N=50) so that the data
    pipeline and masking strategy are identical between dev and final.
    """
    embed_dim = 384
    tokenizer = TokenizerConfig(
        patch_size=50,
        embed_dim=embed_dim,
        kind="linear",
    )
    encoder = EncoderConfig(
        num_leads=NUM_LEADS,
        patch_size=50,
        num_patches=NUM_PATCHES,
        embed_dim=embed_dim,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.1,
        use_flash=True,
        qkv_bias=True,
    )
    predictor = PredictorConfig(
        num_leads=NUM_LEADS,
        num_patches=NUM_PATCHES,
        encoder_embed_dim=embed_dim,
        embed_dim=192,
        depth=3,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.1,
        use_flash=True,
        qkv_bias=True,
    )
    _check_consistency(tokenizer, encoder, predictor)
    return tokenizer, encoder, predictor


def final_preset() -> tuple[TokenizerConfig, EncoderConfig, PredictorConfig]:
    """Paper-faithful architecture for full pretraining runs."""
    embed_dim = 768
    tokenizer = TokenizerConfig(
        patch_size=50,
        embed_dim=embed_dim,
        kind="linear",
    )
    encoder = EncoderConfig(
        num_leads=NUM_LEADS,
        patch_size=50,
        num_patches=NUM_PATCHES,
        embed_dim=embed_dim,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.1,
        use_flash=True,
        qkv_bias=True,
    )
    predictor = PredictorConfig(
        num_leads=NUM_LEADS,
        num_patches=NUM_PATCHES,
        encoder_embed_dim=embed_dim,
        embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.1,
        use_flash=True,
        qkv_bias=True,
    )
    _check_consistency(tokenizer, encoder, predictor)
    return tokenizer, encoder, predictor


def _check_consistency(
    tokenizer: TokenizerConfig,
    encoder: EncoderConfig,
    predictor: PredictorConfig,
) -> None:
    if tokenizer.embed_dim != encoder.embed_dim:
        raise ValueError(
            f"tokenizer.embed_dim ({tokenizer.embed_dim}) must match encoder.embed_dim ({encoder.embed_dim})"
        )
    if encoder.embed_dim != predictor.encoder_embed_dim:
        raise ValueError(
            f"encoder.embed_dim ({encoder.embed_dim}) must match predictor.encoder_embed_dim ({predictor.encoder_embed_dim})"
        )
    if tokenizer.patch_size != encoder.patch_size:
        raise ValueError(
            f"tokenizer.patch_size ({tokenizer.patch_size}) must match encoder.patch_size ({encoder.patch_size})"
        )
    if encoder.num_patches != predictor.num_patches:
        raise ValueError(
            f"encoder.num_patches ({encoder.num_patches}) must match predictor.num_patches ({predictor.num_patches})"
        )
    if encoder.num_leads != predictor.num_leads:
        raise ValueError(
            f"encoder.num_leads ({encoder.num_leads}) must match predictor.num_leads ({predictor.num_leads})"
        )
    if encoder.embed_dim % encoder.num_heads != 0:
        raise ValueError(
            f"encoder.embed_dim ({encoder.embed_dim}) must be divisible by encoder.num_heads ({encoder.num_heads})"
        )
    if predictor.embed_dim % predictor.num_heads != 0:
        raise ValueError(
            f"predictor.embed_dim ({predictor.embed_dim}) must be divisible by predictor.num_heads ({predictor.num_heads})"
        )
    if encoder.embed_dim % 4 != 0:
        raise ValueError(
            f"encoder.embed_dim ({encoder.embed_dim}) must be divisible by 4 for 2D sin/cos positional embeddings"
        )
    if predictor.embed_dim % 2 != 0:
        raise ValueError(
            f"predictor.embed_dim ({predictor.embed_dim}) must be even for 1D sin/cos positional embeddings"
        )
    if not 0.0 <= encoder.drop_path < 1.0:
        raise ValueError(
            f"encoder.drop_path ({encoder.drop_path}) must be in the range [0, 1)"
        )
    if not 0.0 <= predictor.drop_path < 1.0:
        raise ValueError(
            f"predictor.drop_path ({predictor.drop_path}) must be in the range [0, 1)"
        )
