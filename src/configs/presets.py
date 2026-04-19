from .tokenizer_config import TokenizerConfig
from .encoder_config import EncoderConfig
from .predictor_config import PredictorConfig


def dev_preset() -> tuple[TokenizerConfig, EncoderConfig, PredictorConfig]:
    """Smaller architecture for ablations and debugging (~22M encoder, ~5M predictor)."""
    embed_dim = 384
    tokenizer = TokenizerConfig(
        patch_size=50,
        embed_dim=embed_dim,
        conv1_channels=32,
        conv1_kernel=15,
        conv2_channels=64,
        conv2_kernel=9,
    )
    encoder = EncoderConfig(
        num_leads=12,
        patch_size=50,
        num_patches=50,
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
        num_leads=12,
        num_patches=50,
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
    """Full architecture matching the ECG-JEPA paper (~86M encoder, ~22M predictor)."""
    embed_dim = 768
    tokenizer = TokenizerConfig(
        patch_size=50,
        embed_dim=embed_dim,
        conv1_channels=32,
        conv1_kernel=15,
        conv2_channels=64,
        conv2_kernel=9,
    )
    encoder = EncoderConfig(
        num_leads=12,
        patch_size=50,
        num_patches=50,
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
        num_leads=12,
        num_patches=50,
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
