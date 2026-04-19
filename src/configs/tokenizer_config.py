from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    patch_size: int = 50
    embed_dim: int = 384
    conv1_channels: int = 32
    conv1_kernel: int = 15
    conv2_channels: int = 64
    conv2_kernel: int = 9
