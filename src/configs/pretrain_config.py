from dataclasses import dataclass, field


@dataclass
class PretrainConfig:
    """Hyperparameters for JEPA pretraining."""

    # --- Architecture preset ---
    preset: str = "dev"  # "dev" or "final"

    # --- Data ---
    data_dir: str = "data/ptbxl"
    num_workers: int = 4
    signal_length: int = 1000  # T = 1000 for PTB-XL at 100 Hz (10s × 100 Hz)

    # --- Masking ---
    mask_strategy: str = "random"  # "random", "block", "channel"
    mask_ratio_min: float = 0.60
    mask_ratio_max: float = 0.70

    # --- Training ---
    batch_size: int = 64
    num_epochs: int = 100
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    clip_grad: float = 3.0

    # --- EMA ---
    ema_start: float = 0.996
    ema_end: float = 1.0

    # --- Loss ---
    loss_type: str = "l1"  # "l1" or "l2"
    target_layer_norm: bool = True

    # --- Logging & checkpointing ---
    log_every: int = 10
    save_every: int = 20
    save_dir: str = "checkpoints"

    # --- Misc ---
    seed: int = 42
    use_amp: bool = True  # BF16 autocast on activations (weights stay FP32)
    device: str = "cuda"

    # --- Derived (set at runtime) ---
    use_flash: bool = True
