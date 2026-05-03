from dataclasses import dataclass


@dataclass
class EncoderConfig:
    # Defaults follow ECG-JEPA (arxiv 2410.08559):
    #   8 leads (I, II, V1-V6); the 4 derivable leads are dropped via Einthoven's law.
    #   T = 2500 = 10 s @ 250 Hz, patch_size = 50, so num_patches = 50.
    num_leads: int = 8
    patch_size: int = 50
    num_patches: int = 50
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    drop_path: float = 0.1
    use_flash: bool = True
    qkv_bias: bool = True
