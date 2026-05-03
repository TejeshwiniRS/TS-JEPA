from dataclasses import dataclass


@dataclass
class PretrainConfig:
    """Hyperparameters for JEPA pretraining.

    Defaults follow the ECG-JEPA paper (Kim 2026, Tables 11 + 15):
      - 10 s @ 250 Hz signals (T = 2500), 8 leads, patch_size = 50.
      - Random masking ratio sampled uniformly in [0.6, 0.7).
      - L1 loss on representations; no LayerNorm on the teacher target.
      - AdamW, weight_decay 0.05, cosine LR with 5 warmup epochs.
      - EMA momentum 0.996 -> 1.0 over the run.
    """

    # --- Architecture preset ---
    preset: str = "dev"  # "dev" or "final"

    # --- Data ---
    data_dir: str = "data/pretrain"
    num_workers: int = 4
    signal_length: int = 2500  # 10 s @ 250 Hz

    # --- Masking ---
    # "random"      : I-JEPA-style random patch masking.
    # "multi_block" : ECG-JEPA paper's overlapping multi-block masking
    #                 (mask_freq blocks, each of size mask_block_size, allowed
    #                 to overlap; mask_ratio_min/max are ignored in this mode).
    mask_strategy: str = "random"
    mask_ratio_min: float = 0.60
    mask_ratio_max: float = 0.70
    # Multi-block specific:
    mask_freq: int = 4               # number of blocks to draw
    mask_block_size_min: int = 8     # ~ 0.175 * 50 = 8.75 patches @ N=50
    mask_block_size_max: int = 12    # ~ 0.225 * 50 = 11.25 patches @ N=50

    # --- Training ---
    batch_size: int = 128
    num_epochs: int = 100
    lr: float = 2.5e-5             # paper: 2.5e-5 (rb), 5e-5 (mb)
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    clip_grad: float = 3.0

    # --- EMA ---
    ema_start: float = 0.996
    ema_end: float = 1.0

    # --- Loss ---
    loss_type: str = "l1"  # "l1" or "l2"
    # The paper does not LayerNorm the teacher output before the L1 loss.
    # Leaving this on can encourage the teacher to drift toward a low-rank
    # attractor, hiding representation collapse behind a healthy-looking loss.
    target_layer_norm: bool = False

    # --- Logging & checkpointing ---
    log_every: int = 1
    save_every: int = 5
    save_dir: str = "checkpoints"
    metrics_csv: str = "metrics.csv"  # written under save_dir

    # --- Collapse monitoring (eRank + per-token std) ---
    monitor_erank: bool = True
    erank_eps: float = 1e-7
    # Approximate number of tokens / pooled vectors to aggregate for the
    # eRank estimate each epoch. Larger -> lower-variance estimate.
    erank_max_samples: int = 25600
    # Alarm threshold on the *post-pool* eRank (the number that actually
    # matters for downstream linear probes). Below this, the run is logged
    # as "COLLAPSE" in the metrics CSV (no auto-stop, just visibility).
    erank_alarm_post_pool: float = 8.0

    # --- Misc ---
    seed: int = 42
    use_amp: bool = True  # BF16 autocast on activations (weights stay FP32)
    device: str = "cuda"

    # --- Derived (set at runtime) ---
    use_flash: bool = True
