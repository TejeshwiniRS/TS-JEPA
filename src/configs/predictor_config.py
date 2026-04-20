from dataclasses import dataclass


@dataclass
class PredictorConfig:
    num_leads: int = 12
    num_patches: int = 20
    encoder_embed_dim: int = 384
    embed_dim: int = 192
    depth: int = 3
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.1
    use_flash: bool = True
    qkv_bias: bool = True
