from dataclasses import dataclass


@dataclass
class PredictorConfig:
    # Defaults match ECG-JEPA: 8 leads, N = 50 temporal patches.
    num_leads: int = 8
    num_patches: int = 50
    encoder_embed_dim: int = 384
    embed_dim: int = 192
    depth: int = 3
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.1
    use_flash: bool = True
    qkv_bias: bool = True
