from dataclasses import dataclass


@dataclass
class EncoderConfig:
    num_leads: int = 12
    patch_size: int = 50
    num_patches: int = 20
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.1
    use_flash: bool = True
    qkv_bias: bool = True
