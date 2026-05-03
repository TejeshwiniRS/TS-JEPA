from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    """Per-patch tokenizer.

    Two implementations are supported, selected by `kind`:
      - "linear": single Linear(patch_size -> embed_dim). This matches the
        ECG-JEPA paper, which describes patches as being projected via "a
        linear layer" before adding positional embeddings.
      - "ffn":    Linear -> GELU -> Linear with `ffn_hidden_dim` in the middle.
        Slightly more expressive; useful for ablations.
    """

    patch_size: int = 50
    embed_dim: int = 384
    kind: str = "linear"
    ffn_hidden_dim: int = 256  # only used when kind == "ffn"
