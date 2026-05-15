"""Generic train + eval loop, model-agnostic.

Used by all three notebooks. The notebooks build the model + optimizer +
scheduler themselves (they differ across architectures); here we provide
just the loop and metric reporting.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics


@dataclass
class FitHistory:
    train_loss: list[float] = field(default_factory=list)
    val_bacc:   list[float] = field(default_factory=list)
    val_wf1:    list[float] = field(default_factory=list)
    val_kappa:  list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_bacc: float = -1.0
    best_state_dict: dict | None = None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Return metrics dict + raw preds/targets."""
    model.eval()
    all_preds, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        all_targets.append(y)
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    out = compute_metrics(targets, preds)
    out["preds"] = preds
    out["targets"] = targets
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> FitHistory:
    """Train `epochs` epochs, tracking best val balanced-accuracy."""
    hist = FitHistory()
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                  device, grad_clip=grad_clip)
        if scheduler is not None:
            scheduler.step()
        val = evaluate(model, val_loader, device)

        hist.train_loss.append(tr_loss)
        hist.val_bacc.append(val["bacc"])
        hist.val_wf1.append(val["wf1"])
        hist.val_kappa.append(val["kappa"])

        if val["bacc"] > hist.best_val_bacc:
            hist.best_val_bacc = val["bacc"]
            hist.best_epoch = ep
            hist.best_state_dict = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )

        if verbose:
            lr = optimizer.param_groups[0]["lr"]
            print(f"epoch {ep:3d} | train_loss {tr_loss:.4f} | "
                  f"val bacc {val['bacc']*100:5.2f} wf1 {val['wf1']*100:5.2f} "
                  f"kappa {val['kappa']*100:5.2f} | lr {lr:.2e}", flush=True)

    if verbose:
        print(f"\nbest val bacc {hist.best_val_bacc*100:.2f} @ epoch {hist.best_epoch}")
    return hist


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> torch.optim.lr_scheduler.SequentialLR:
    """Linear warmup → cosine anneal. Step once per epoch."""
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
