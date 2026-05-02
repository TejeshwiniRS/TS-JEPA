from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    patch_size: int = 50
    embed_dim: int = 384
    ffn_hidden_dim: int = 256
