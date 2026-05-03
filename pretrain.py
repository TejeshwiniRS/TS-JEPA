"""JEPA pretraining script for multivariate time series (12-lead ECG).

Defaults follow the ECG-JEPA paper (Kim 2026, arxiv 2410.08559) but with
**full self-attention instead of CroPA** by deliberate choice:

  - 8 leads (I, II, V1-V6), 10 s @ 250 Hz, T = 2500, patch_size = 50, N = 50.
  - Random masking ratio in [0.6, 0.7), or paper-faithful overlapping
    multi-block masking via ``--mask_strategy multi_block``.
  - L1 loss between predictor outputs and target encoder outputs at masked
    positions, **with no LayerNorm on the teacher target** (the LN trick the
    previous version applied is what hides representation collapse from the
    loss curve).
  - AdamW, weight_decay = 0.05, cosine LR with 5 warmup epochs.
  - EMA momentum 0.996 -> 1.0 over the run.

Representation-collapse monitoring (eRank) is computed every epoch on a
fixed sample of validation signals, on three views:

  * ``erank_ctx_token``   — context (student) encoder, per-token features.
  * ``erank_tgt_token``   — target  (teacher) encoder, per-token features.
  * ``erank_ctx_pool``    — context encoder output average-pooled per sample
                            (this is exactly the vector a downstream linear
                            probe would receive). **Most diagnostic** of the
                            three; if this drops near 1, downstream tasks
                            are dead even when train_loss looks fine.

All metrics are appended to ``<save_dir>/metrics.csv`` for after-the-fact
plotting.

Example::

    python -u pretrain.py \\
        --data_dir data/pretrain --preset final \\
        --mask_strategy random --batch_size 128 --lr 2.5e-5 \\
        --num_epochs 100 --save_dir checkpoints/final_rb
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import dev_preset, final_preset
from src.configs.pretrain_config import PretrainConfig
from src.data.pretrain_dataset import get_pretrain_loaders
from src.encoder import ECGEncoder
from src.masking import get_mask_fn
from src.predictor import ECGPredictor
from src.tokenizer import ECGTokenizer


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def amp_autocast(device: torch.device, use_amp: bool):
    """BF16 autocast on CUDA, no-op otherwise.

    Weights stay FP32 - only activations are cast to BF16 during the forward
    pass. BF16 has the same exponent range as FP32, so GradScaler is not
    needed (unlike FP16).
    """
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullCtx()


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: int = 0,
    warmup_start_value: float = 0.0,
) -> list[float]:
    warmup = []
    if warmup_epochs > 0:
        warmup = [
            warmup_start_value + i * (base_value - warmup_start_value) / warmup_epochs
            for i in range(warmup_epochs)
        ]
    cos_epochs = max(1, epochs - warmup_epochs)
    cos = [
        final_value
        + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / cos_epochs))
        for i in range(cos_epochs)
    ]
    return warmup + cos


def ema_momentum_schedule(
    ema_start: float,
    ema_end: float,
    num_epochs: int,
) -> list[float]:
    return [
        ema_end - 0.5 * (ema_end - ema_start) * (1 + math.cos(math.pi * i / num_epochs))
        for i in range(num_epochs)
    ]


# ---------------------------------------------------------------------------
# EMA / loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def ema_update(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    momentum: float,
) -> None:
    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_tgt.data.mul_(momentum).add_(p_ctx.detach().data, alpha=1.0 - momentum)


def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        return (preds - targets).abs().mean()
    if loss_type == "l2":
        return ((preds - targets) ** 2).mean()
    raise ValueError(f"Unknown loss_type '{loss_type}'")


# ---------------------------------------------------------------------------
# Effective rank
# ---------------------------------------------------------------------------

def effective_rank(Z: torch.Tensor, eps: float = 1e-8) -> float:
    """Effective rank of a representation matrix ``Z in R^{N x D}``.

    Defined as ``exp(H(p))`` where ``p`` are the normalized eigenvalues of the
    centered covariance ``Z^T Z / (N - 1)``. Range: ``[1, D]``. Healthy SSL
    runs sit in the tens to hundreds; collapse drives the value toward 1.
    """
    Z = Z.float()
    if Z.size(0) <= 1:
        return 1.0
    Z = Z - Z.mean(dim=0, keepdim=True)
    cov = (Z.T @ Z) / (Z.size(0) - 1)
    # `eigvalsh` is numerically stable for symmetric matrices.
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eigvals.sum()
    if total < eps:
        return 1.0
    p = eigvals / total
    H = -(p * (p + eps).log()).sum()
    return float(H.exp().item())


@torch.no_grad()
def measure_collapse(
    context_encoder: ECGEncoder,
    target_encoder: ECGEncoder,
    tokenizer: ECGTokenizer,
    sample_signals: torch.Tensor,
    device: torch.device,
    use_amp: bool,
    erank_eps: float,
) -> dict[str, float]:
    """Compute eRank + std diagnostics on a fixed batch of signals.

    Three eRanks are reported:

      - ``erank_ctx_token`` over the context encoder's per-patch tokens
        (concatenated across the sample).
      - ``erank_tgt_token`` same for the target encoder.
      - ``erank_ctx_pool``  over the **average-pooled** context encoder
        output, one D-vector per sample. This is the representation that a
        downstream linear probe consumes; its eRank is the single number
        most predictive of downstream-task viability.

    Per-token std is also tracked because it is a faster early warning than
    eRank when collapse is happening fast.
    """
    context_encoder.eval()
    target_encoder.eval()

    sig = sample_signals.to(device, non_blocking=True)
    patches = tokenizer.patchify(sig)
    with amp_autocast(device, use_amp):
        ctx = context_encoder.forward_all(patches).float()  # (bs, C*N, D)
        tgt = target_encoder.forward_all(patches).float()   # (bs, C*N, D)

    bs, cn, d = ctx.shape
    ctx_pool = ctx.mean(dim=1)  # (bs, D) -- the linear-probe input

    # Per-token std (averaged over tokens then samples) - cheap collapse alarm.
    ctx_token_std = float(ctx.std(dim=1).mean().item())
    tgt_token_std = float(tgt.std(dim=1).mean().item())
    ctx_pool_std = float(ctx_pool.std(dim=0).mean().item())

    return {
        "erank_ctx_token": effective_rank(ctx.reshape(-1, d), eps=erank_eps),
        "erank_tgt_token": effective_rank(tgt.reshape(-1, d), eps=erank_eps),
        "erank_ctx_pool": effective_rank(ctx_pool, eps=erank_eps),
        "std_ctx_token": ctx_token_std,
        "std_tgt_token": tgt_token_std,
        "std_ctx_pool": ctx_pool_std,
    }


def build_collapse_probe_signals(
    val_loader,
    max_samples: int,
) -> torch.Tensor:
    """Concatenate up to `max_samples` signals from the val loader (CPU).

    We fix the probe set once at the start of training so eRank trajectories
    are comparable across epochs.
    """
    chunks: list[torch.Tensor] = []
    collected = 0
    for sig in val_loader:
        if isinstance(sig, (list, tuple)):
            sig = sig[0]
        chunks.append(sig)
        collected += sig.size(0)
        if collected >= max_samples:
            break
    out = torch.cat(chunks, dim=0)
    if out.size(0) > max_samples:
        out = out[:max_samples]
    return out


# ---------------------------------------------------------------------------
# Checkpoint / metrics IO
# ---------------------------------------------------------------------------

def save_checkpoint(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    cfg: PretrainConfig,
    path: str,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "context_encoder": context_encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )


METRICS_FIELDS: list[str] = [
    "epoch", "lr", "ema_m",
    "train_loss", "val_loss",
    "erank_ctx_token", "erank_tgt_token", "erank_ctx_pool",
    "std_ctx_token", "std_tgt_token", "std_ctx_pool",
    "collapse_alarm",
    "elapsed_s",
]


def append_metrics(path: Path, row: dict) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in METRICS_FIELDS})


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training / validation step
# ---------------------------------------------------------------------------

def _sample_mask(mask_fn, cfg: PretrainConfig, num_patches: int, device: torch.device):
    if cfg.mask_strategy == "multi_block":
        # Driving dim is freq + block size; we still pass a `mask_ratio` but
        # the function ignores it when block_size_min/max are supplied.
        ratio = (cfg.mask_ratio_min + cfg.mask_ratio_max) / 2
        return mask_fn(
            num_patches=num_patches,
            mask_ratio=ratio,
            device=device,
            freq=cfg.mask_freq,
            block_size_min=cfg.mask_block_size_min,
            block_size_max=cfg.mask_block_size_max,
        )
    ratio = (
        cfg.mask_ratio_min
        + (cfg.mask_ratio_max - cfg.mask_ratio_min) * torch.rand(1).item()
    )
    return mask_fn(num_patches=num_patches, mask_ratio=ratio, device=device)


def train_one_epoch(
    context_encoder: ECGEncoder,
    target_encoder: ECGEncoder,
    predictor: ECGPredictor,
    tokenizer: ECGTokenizer,
    train_loader,
    optimizer: torch.optim.Optimizer,
    mask_fn,
    cfg: PretrainConfig,
    enc_cfg,
    epoch: int,
    lr: float,
    ema_m: float,
    device: torch.device,
) -> dict[str, float]:
    context_encoder.train()
    predictor.train()
    target_encoder.eval()

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    total_loss = 0.0
    num_batches = 0

    for signal in train_loader:
        if isinstance(signal, (list, tuple)):
            signal = signal[0]
        signal = signal.to(device, non_blocking=True)  # (bs, C, T)
        patches = tokenizer.patchify(signal)            # (bs, C, N, P)

        vis_idx, msk_idx = _sample_mask(mask_fn, cfg, enc_cfg.num_patches, device)

        optimizer.zero_grad(set_to_none=True)

        with amp_autocast(device, cfg.use_amp):
            ctx_tokens = context_encoder(patches, visible_indices=vis_idx)
            preds = predictor(ctx_tokens, vis_idx, msk_idx)

            with torch.no_grad():
                tgt_all = target_encoder.forward_all(patches)
                if cfg.target_layer_norm:
                    tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
                tgt_all = tgt_all.reshape(
                    signal.size(0), enc_cfg.num_leads, enc_cfg.num_patches, -1
                )
                targets = tgt_all.index_select(dim=2, index=msk_idx)

            loss = compute_loss(preds, targets, cfg.loss_type)

        loss.backward()

        if cfg.clip_grad > 0:
            nn.utils.clip_grad_norm_(
                list(context_encoder.parameters()) + list(predictor.parameters()),
                cfg.clip_grad,
            )
        optimizer.step()
        ema_update(context_encoder, target_encoder, ema_m)

        total_loss += float(loss.item())
        num_batches += 1

    return {"train_loss": total_loss / max(num_batches, 1)}


@torch.no_grad()
def validate(
    context_encoder: ECGEncoder,
    target_encoder: ECGEncoder,
    predictor: ECGPredictor,
    tokenizer: ECGTokenizer,
    val_loader,
    mask_fn,
    cfg: PretrainConfig,
    enc_cfg,
    device: torch.device,
) -> dict[str, float]:
    context_encoder.eval()
    predictor.eval()
    target_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    for signal in val_loader:
        if isinstance(signal, (list, tuple)):
            signal = signal[0]
        signal = signal.to(device, non_blocking=True)
        patches = tokenizer.patchify(signal)

        vis_idx, msk_idx = _sample_mask(mask_fn, cfg, enc_cfg.num_patches, device)

        with amp_autocast(device, cfg.use_amp):
            ctx_tokens = context_encoder(patches, visible_indices=vis_idx)
            preds = predictor(ctx_tokens, vis_idx, msk_idx)

            tgt_all = target_encoder.forward_all(patches)
            if cfg.target_layer_norm:
                tgt_all = F.layer_norm(tgt_all, (tgt_all.size(-1),))
            tgt_all = tgt_all.reshape(
                signal.size(0), enc_cfg.num_leads, enc_cfg.num_patches, -1
            )
            targets = tgt_all.index_select(dim=2, index=msk_idx)
            loss = compute_loss(preds, targets, cfg.loss_type)

        total_loss += float(loss.item())
        num_batches += 1

    return {"val_loss": total_loss / max(num_batches, 1)}


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JEPA pretraining for ECG time series")
    p.add_argument("--preset", type=str, default="dev", choices=["dev", "final"])
    p.add_argument("--data_dir", type=str, default="data/pretrain")
    p.add_argument("--signal_length", type=int, default=2500)

    # Masking
    p.add_argument(
        "--mask_strategy", type=str, default="random",
        choices=["random", "block", "multi_block"],
    )
    p.add_argument("--mask_ratio_min", type=float, default=0.60)
    p.add_argument("--mask_ratio_max", type=float, default=0.70)
    p.add_argument("--mask_freq", type=int, default=4)
    p.add_argument("--mask_block_size_min", type=int, default=8)
    p.add_argument("--mask_block_size_max", type=int, default=12)

    # Training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)

    # Loss
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument(
        "--target_layer_norm", action="store_true", default=False,
        help="Apply LayerNorm to teacher output before the loss. Off by "
             "default to match the paper; turning it on can hide collapse.",
    )

    # Logging
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--erank_max_samples", type=int, default=4096)
    p.add_argument("--erank_alarm_post_pool", type=float, default=8.0)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = PretrainConfig(
        preset=args.preset,
        data_dir=args.data_dir,
        signal_length=args.signal_length,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        clip_grad=args.clip_grad,
        ema_start=args.ema_start,
        ema_end=args.ema_end,
        mask_strategy=args.mask_strategy,
        mask_ratio_min=args.mask_ratio_min,
        mask_ratio_max=args.mask_ratio_max,
        mask_freq=args.mask_freq,
        mask_block_size_min=args.mask_block_size_min,
        mask_block_size_max=args.mask_block_size_max,
        loss_type=args.loss_type,
        target_layer_norm=args.target_layer_norm,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        erank_max_samples=args.erank_max_samples,
        erank_alarm_post_pool=args.erank_alarm_post_pool,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"
        cfg.use_amp = False
    device = torch.device(cfg.device)

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    preset_fn = {"dev": dev_preset, "final": final_preset}[cfg.preset]
    tok_cfg, enc_cfg, pred_cfg = preset_fn()

    if device.type == "cpu":
        enc_cfg.use_flash = False
        pred_cfg.use_flash = False
    cfg.use_flash = enc_cfg.use_flash

    assert cfg.signal_length % enc_cfg.patch_size == 0, (
        f"signal_length ({cfg.signal_length}) must be divisible by "
        f"patch_size ({enc_cfg.patch_size})"
    )
    expected_patches = cfg.signal_length // enc_cfg.patch_size
    assert expected_patches == enc_cfg.num_patches, (
        f"signal_length / patch_size = {expected_patches} but "
        f"enc_cfg.num_patches = {enc_cfg.num_patches}. "
        "Adjust signal_length or num_patches."
    )

    print("=" * 70)
    print("TS-JEPA Pretraining (full attention; no CroPA)")
    print("=" * 70)
    print(f"  Preset:          {cfg.preset}")
    print(f"  Device:          {device}")
    print(f"  AMP (BF16):      {cfg.use_amp and device.type == 'cuda'}")
    print(f"  Flash Attention: {cfg.use_flash}")
    print(f"  Tokenizer:       {tok_cfg.kind}")
    print(f"  Mask strategy:   {cfg.mask_strategy}", end="")
    if cfg.mask_strategy == "multi_block":
        print(
            f"  (freq={cfg.mask_freq}, "
            f"block_size=[{cfg.mask_block_size_min},{cfg.mask_block_size_max}])"
        )
    else:
        print(f"  (ratio=[{cfg.mask_ratio_min:.2f},{cfg.mask_ratio_max:.2f}))")
    print(f"  Loss:            {cfg.loss_type}  (target_LN={cfg.target_layer_norm})")
    print(f"  Epochs:          {cfg.num_epochs}")
    print(f"  Batch size:      {cfg.batch_size}")
    print(f"  LR:              {cfg.lr} -> {cfg.min_lr} (warmup={cfg.warmup_epochs})")
    print(f"  EMA momentum:    {cfg.ema_start} -> {cfg.ema_end}")
    print(
        f"  Signal:          C={enc_cfg.num_leads}, T={cfg.signal_length}, "
        f"N={enc_cfg.num_patches}, P={enc_cfg.patch_size}"
    )
    print()

    train_loader, val_loader = get_pretrain_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"  Train samples:   {len(train_loader.dataset)}")
    print(f"  Val samples:     {len(val_loader.dataset)}")

    tokenizer = ECGTokenizer(tok_cfg)
    context_encoder = ECGEncoder(enc_cfg, tokenizer).to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    predictor = ECGPredictor(pred_cfg).to(device)

    print(f"  Encoder params:  {count_parameters(context_encoder):,}")
    print(f"  Predictor params:{count_parameters(predictor):,}")
    print("=" * 70)

    optimizer = torch.optim.AdamW(
        [
            {"params": context_encoder.parameters()},
            {"params": predictor.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    lr_schedule = cosine_scheduler(
        base_value=cfg.lr,
        final_value=cfg.min_lr,
        epochs=cfg.num_epochs,
        warmup_epochs=cfg.warmup_epochs,
        warmup_start_value=1e-6,
    )
    momentum_schedule = ema_momentum_schedule(
        cfg.ema_start, cfg.ema_end, cfg.num_epochs
    )

    mask_fn = get_mask_fn(cfg.mask_strategy)

    start_epoch = 0
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        target_encoder.load_state_dict(ckpt["target_encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    save_checkpoint(
        context_encoder, target_encoder, predictor, optimizer,
        epoch=0, loss=float("inf"), cfg=cfg,
        path=os.path.join(cfg.save_dir, "checkpoint_epoch_0.pt"),
    )

    # Fix the eRank probe set once so trajectories are comparable across epochs.
    probe_signals = build_collapse_probe_signals(val_loader, cfg.erank_max_samples)
    print(
        f"  Collapse probe:  {probe_signals.shape[0]} samples "
        f"({probe_signals.shape[1]} leads x {probe_signals.shape[2]} samples each)"
    )

    metrics_path = Path(cfg.save_dir) / cfg.metrics_csv
    print(f"  Metrics CSV:     {metrics_path}")
    print("=" * 70)

    best_val_loss = float("inf")
    val_metrics: dict[str, float] = {"val_loss": float("nan")}

    for epoch in range(start_epoch, cfg.num_epochs):
        t0 = time.time()
        lr = lr_schedule[epoch]
        ema_m = momentum_schedule[epoch]

        train_metrics = train_one_epoch(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            mask_fn=mask_fn,
            cfg=cfg,
            enc_cfg=enc_cfg,
            epoch=epoch,
            lr=lr,
            ema_m=ema_m,
            device=device,
        )

        val_metrics = validate(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            tokenizer=tokenizer,
            val_loader=val_loader,
            mask_fn=mask_fn,
            cfg=cfg,
            enc_cfg=enc_cfg,
            device=device,
        )

        coll = measure_collapse(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            tokenizer=tokenizer,
            sample_signals=probe_signals,
            device=device,
            use_amp=cfg.use_amp,
            erank_eps=cfg.erank_eps,
        )

        elapsed = time.time() - t0
        alarm = coll["erank_ctx_pool"] < cfg.erank_alarm_post_pool

        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            print(
                f"Epoch {epoch:4d}/{cfg.num_epochs} | "
                f"lr {lr:.2e} | ema_m {ema_m:.5f} | "
                f"train_loss {train_metrics['train_loss']:.4f} | "
                f"val_loss {val_metrics['val_loss']:.4f} | "
                f"erank ctx_tok {coll['erank_ctx_token']:.1f} "
                f"tgt_tok {coll['erank_tgt_token']:.1f} "
                f"ctx_pool {coll['erank_ctx_pool']:.1f}"
                f"{' <- COLLAPSE' if alarm else ''} | "
                f"std ctx_tok {coll['std_ctx_token']:.3f} "
                f"ctx_pool {coll['std_ctx_pool']:.3f} | "
                f"{elapsed:.1f}s"
            )

        append_metrics(metrics_path, {
            "epoch": epoch,
            "lr": f"{lr:.6e}",
            "ema_m": f"{ema_m:.6f}",
            "train_loss": f"{train_metrics['train_loss']:.6f}",
            "val_loss": f"{val_metrics['val_loss']:.6f}",
            "erank_ctx_token": f"{coll['erank_ctx_token']:.4f}",
            "erank_tgt_token": f"{coll['erank_tgt_token']:.4f}",
            "erank_ctx_pool": f"{coll['erank_ctx_pool']:.4f}",
            "std_ctx_token": f"{coll['std_ctx_token']:.6f}",
            "std_tgt_token": f"{coll['std_tgt_token']:.6f}",
            "std_ctx_pool": f"{coll['std_ctx_pool']:.6f}",
            "collapse_alarm": int(alarm),
            "elapsed_s": f"{elapsed:.2f}",
        })

        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(
                context_encoder, target_encoder, predictor, optimizer,
                epoch=epoch, loss=val_metrics["val_loss"], cfg=cfg,
                path=ckpt_path,
            )
            print(f"  -> Saved checkpoint: {ckpt_path}")

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                context_encoder, target_encoder, predictor, optimizer,
                epoch=epoch, loss=best_val_loss, cfg=cfg,
                path=os.path.join(cfg.save_dir, "checkpoint_best.pt"),
            )

    save_checkpoint(
        context_encoder, target_encoder, predictor, optimizer,
        epoch=cfg.num_epochs - 1, loss=val_metrics["val_loss"], cfg=cfg,
        path=os.path.join(cfg.save_dir, "checkpoint_final.pt"),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {cfg.save_dir}/")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
