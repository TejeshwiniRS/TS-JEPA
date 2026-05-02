"""JEPA pretraining script for multivariate time series (12-lead ECG).

Usage:
    # Dev pretraining on PTB-XL
    python pretrain.py --data_dir data/ptbxl --preset dev --num_epochs 100

    # Full-scale pretraining
    python pretrain.py --data_dir data/ptbxl --preset final --num_epochs 100

Architecture flow per training step:
    1. Sample mask → (visible_indices [Q], masked_indices [M])
    2. Context encoder: patches → (bs, C×Q, D)        [gradient flows]
    3. Predictor:      (bs×C, Q, D) → (bs, C, M, D)   [gradient flows]
    4. Target encoder:  patches → (bs, C×N, D)         [no gradient, EMA]
    5. Loss: L1 between predictor output and target at masked positions
    6. EMA update of target encoder parameters
    
    
    python -u pretrain.py --data_dir /home/aimakeradmin/shady/TS-JEPA/data/mimic --preset final --batch_size 512 --num_epochs 100 --num_workers 12 --save_dir /home/aimakeradmin/shady/TS-JEPA/checkpoints/mimic
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import dev_preset, final_preset
from src.configs.pretrain_config import PretrainConfig
from src.tokenizer import ECGTokenizer
from src.encoder import ECGEncoder
from src.predictor import ECGPredictor
from src.masking import get_mask_fn
from src.data.ptbxl_dataset import get_pretrain_loaders


class _NullCtx:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def amp_autocast(device: torch.device, use_amp: bool):
    """BF16 autocast on CUDA, no-op otherwise.

    Weights stay FP32 — only activations are cast to BF16 during the forward
    pass. BF16 has the same exponent range as FP32, so GradScaler is not
    needed (unlike FP16).
    """
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return _NullCtx()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    warmup_epochs: int = 0,
    warmup_start_value: float = 0.0,
) -> list[float]:
    """Cosine schedule with linear warmup. Returns one value per epoch."""
    warmup_schedule = []
    if warmup_epochs > 0:
        warmup_schedule = [
            warmup_start_value + i * (base_value - warmup_start_value) / warmup_epochs
            for i in range(warmup_epochs)
        ]
    cos_epochs = epochs - warmup_epochs
    cos_schedule = [
        final_value
        + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / cos_epochs))
        for i in range(cos_epochs)
    ]
    return warmup_schedule + cos_schedule


def ema_momentum_schedule(
    ema_start: float,
    ema_end: float,
    num_epochs: int,
) -> list[float]:
    """Cosine schedule for EMA momentum from ema_start → ema_end."""
    return [
        ema_end - 0.5 * (ema_end - ema_start) * (1 + math.cos(math.pi * i / num_epochs))
        for i in range(num_epochs)
    ]


@torch.no_grad()
def ema_update(
    context_encoder: nn.Module,
    target_encoder: nn.Module,
    momentum: float,
) -> None:
    """Exponential moving average update of the target encoder."""
    for p_ctx, p_tgt in zip(context_encoder.parameters(), target_encoder.parameters()):
        p_tgt.data.mul_(momentum).add_(p_ctx.detach().data, alpha=1.0 - momentum)


def compute_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Loss between predictor outputs and target encoder outputs at masked positions.

    Args:
        preds: (bs, C, M, D)
        targets: (bs, C, M, D)
        loss_type: "l1" or "l2"
    """
    if loss_type == "l1":
        return (preds - targets).abs().mean()
    elif loss_type == "l2":
        return ((preds - targets) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'")


def effective_rank(Z: torch.Tensor, eps: float = 1e-8) -> float:
    """eRank of representation matrix Z ∈ ℝ^{N×D}.

    Returns exp(entropy of normalised eigenvalue spectrum of the covariance).
    Range: [1, D]. Collapse → 1; fully spread → D.
    """
    Z = Z.float()
    Z = Z - Z.mean(dim=0, keepdim=True)
    N = Z.size(0)
    cov = (Z.T @ Z) / max(N - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    total = eigvals.sum()
    if total < eps:
        return 1.0
    p = eigvals / total
    H = -(p * (p + eps).log()).sum()
    return H.exp().item()


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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

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
        signal = signal.to(device, non_blocking=True)  # (bs, C, T)
        patches = tokenizer.patchify(signal)  # (bs, C, N, patch_size)

        mask_ratio = (
            cfg.mask_ratio_min
            + (cfg.mask_ratio_max - cfg.mask_ratio_min) * torch.rand(1).item()
        )
        vis_idx, msk_idx = mask_fn(
            num_patches=enc_cfg.num_patches,
            mask_ratio=mask_ratio,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        # BF16 autocast: activations are cast to bfloat16 for speed,
        # but weights and gradients stay in FP32. No GradScaler needed
        # because BF16 has the same exponent range as FP32.
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

        # EMA runs in FP32 on the FP32 weights — outside autocast.
        ema_update(context_encoder, target_encoder, ema_m)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"train_loss": avg_loss}


# ---------------------------------------------------------------------------
# Validation step
# ---------------------------------------------------------------------------

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
        signal = signal.to(device, non_blocking=True)
        patches = tokenizer.patchify(signal)

        mask_ratio = (cfg.mask_ratio_min + cfg.mask_ratio_max) / 2
        vis_idx, msk_idx = mask_fn(
            num_patches=enc_cfg.num_patches,
            mask_ratio=mask_ratio,
            device=device,
        )

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

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return {"val_loss": avg_loss}


# ---------------------------------------------------------------------------
# Representation collapse monitor
# ---------------------------------------------------------------------------

@torch.no_grad()
def check_collapse(
    target_encoder: ECGEncoder,
    tokenizer: ECGTokenizer,
    signal_batch: torch.Tensor,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """Compute std of target representations to detect collapse.

    If all representations converge to the same vector, std → 0.
    Healthy training keeps std around 0.5–1.0.
    """
    patches = tokenizer.patchify(signal_batch)
    with amp_autocast(device, use_amp):
        tgt_all = target_encoder.forward_all(patches)
    tgt_float = tgt_all.float()  # (bs, C*N, D)
    token_std = tgt_float.std(dim=1).mean().item()
    erank = effective_rank(tgt_float.reshape(-1, tgt_float.size(-1)))
    return {"repr_std": token_std, "erank": erank}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="JEPA pretraining for ECG time series")
    p.add_argument("--preset", type=str, default="dev", choices=["dev", "final"])
    p.add_argument("--data_dir", type=str, default="data/ptbxl")
    p.add_argument("--signal_length", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--clip_grad", type=float, default=3.0)
    p.add_argument("--ema_start", type=float, default=0.996)
    p.add_argument("--ema_end", type=float, default=1.0)
    p.add_argument("--mask_strategy", type=str, default="block", choices=["random", "block"])
    p.add_argument("--mask_ratio_min", type=float, default=0.60)
    p.add_argument("--mask_ratio_max", type=float, default=0.70)
    p.add_argument("--loss_type", type=str, default="l1", choices=["l1", "l2"])
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=2)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Build config from CLI args.
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
        loss_type=args.loss_type,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=args.use_amp,
    )

    # Device setup.
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        cfg.device = "cpu"
        cfg.use_amp = False
    device = torch.device(cfg.device)

    # Reproducibility.
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Architecture configs.
    preset_map = {"dev": dev_preset, "final": final_preset}
    preset_fn = preset_map[cfg.preset]
    tok_cfg, enc_cfg, pred_cfg = preset_fn()

    # Override flash attention based on device.
    if device.type == "cpu":
        enc_cfg.use_flash = False
        pred_cfg.use_flash = False
    cfg.use_flash = enc_cfg.use_flash

    # Verify signal length is compatible with patch_size.
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
    print("TS-JEPA Pretraining")
    print("=" * 70)
    print(f"  Preset:          {cfg.preset}")
    print(f"  Device:          {device}")
    print(f"  AMP (BF16):      {cfg.use_amp and device.type == 'cuda'}")
    print(f"  Flash Attention: {cfg.use_flash}")
    print(f"  Mask strategy:   {cfg.mask_strategy}")
    print(f"  Mask ratio:      [{cfg.mask_ratio_min:.2f}, {cfg.mask_ratio_max:.2f})")
    print(f"  Loss:            {cfg.loss_type}")
    print(f"  Epochs:          {cfg.num_epochs}")
    print(f"  Batch size:      {cfg.batch_size}")
    print(f"  LR:              {cfg.lr} → {cfg.min_lr} (warmup={cfg.warmup_epochs})")
    print(f"  EMA momentum:    {cfg.ema_start} → {cfg.ema_end}")
    print(f"  Signal length:   {cfg.signal_length} (N={enc_cfg.num_patches}, ps={enc_cfg.patch_size})")
    print()

    # Data.
    train_loader, val_loader = get_pretrain_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    print(f"  Train samples:   {len(train_loader.dataset)}")
    print(f"  Val samples:     {len(val_loader.dataset)}")

    # Build models.
    tokenizer = ECGTokenizer(tok_cfg)
    context_encoder = ECGEncoder(enc_cfg, tokenizer).to(device)
    target_encoder = copy.deepcopy(context_encoder).to(device)
    for p in target_encoder.parameters():
        p.requires_grad_(False)
    predictor = ECGPredictor(pred_cfg).to(device)

    print(f"  Encoder params:  {count_parameters(context_encoder):,}")
    print(f"  Predictor params:{count_parameters(predictor):,}")
    print("=" * 70)

    # Optimizer — AdamW with separate param groups.
    optimizer = torch.optim.AdamW(
        [
            {"params": context_encoder.parameters()},
            {"params": predictor.parameters()},
        ],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    # Schedules.
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

    # Resume from checkpoint.
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        context_encoder.load_state_dict(ckpt["context_encoder"])
        target_encoder.load_state_dict(ckpt["target_encoder"])
        predictor.load_state_dict(ckpt["predictor"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resumed at epoch {start_epoch}")

    # Save initial checkpoint (epoch 0).
    save_checkpoint(
        context_encoder, target_encoder, predictor, optimizer,
        epoch=0, loss=float("inf"), cfg=cfg,
        path=os.path.join(cfg.save_dir, "checkpoint_epoch_0.pt"),
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")

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

        # Collapse check — use first batch of validation data.
        collapse_signal = next(iter(val_loader)).to(device)
        collapse_metrics = check_collapse(target_encoder, tokenizer, collapse_signal, device, cfg.use_amp)

        elapsed = time.time() - t0

        if epoch % cfg.log_every == 0 or epoch == cfg.num_epochs - 1:
            print(
                f"Epoch {epoch:4d}/{cfg.num_epochs} | "
                f"lr {lr:.2e} | ema_m {ema_m:.5f} | "
                f"train_loss {train_metrics['train_loss']:.4f} | "
                f"val_loss {val_metrics['val_loss']:.4f} | "
                f"repr_std {collapse_metrics['repr_std']:.4f} | "
                f"erank {collapse_metrics['erank']:.1f} | "
                f"{elapsed:.1f}s"
            )

        # Save periodic checkpoint.
        if (epoch + 1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(
                context_encoder, target_encoder, predictor, optimizer,
                epoch=epoch, loss=val_metrics["val_loss"], cfg=cfg,
                path=ckpt_path,
            )
            print(f"  → Saved checkpoint: {ckpt_path}")

        # Save best model.
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            save_checkpoint(
                context_encoder, target_encoder, predictor, optimizer,
                epoch=epoch, loss=best_val_loss, cfg=cfg,
                path=os.path.join(cfg.save_dir, "checkpoint_best.pt"),
            )

    # Save final checkpoint.
    save_checkpoint(
        context_encoder, target_encoder, predictor, optimizer,
        epoch=cfg.num_epochs - 1, loss=val_metrics["val_loss"], cfg=cfg,
        path=os.path.join(cfg.save_dir, "checkpoint_final.pt"),
    )
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {cfg.save_dir}/")


if __name__ == "__main__":
    main()
