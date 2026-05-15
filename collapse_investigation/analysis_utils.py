from __future__ import annotations

import csv
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


RUN_LABELS = {
    "final_preset_final": "Final preset, 8 leads, 250 Hz",
    "dev_preset_final": "Dev preset, 8 leads, 250 Hz",
    "dev_100hz": "Dev preset, 8 leads, 100 Hz",
    "dev_100hz_ffn": "Dev preset, FFN tokenizer, 8 leads, 100 Hz",
    "dev_12lead": "Dev preset, 12 leads, 250 Hz",
}

RUN_COLORS = {
    "final_preset_final": "#1f77b4",
    "dev_preset_final": "#2ca02c",
    "dev_100hz": "#d62728",
    "dev_100hz_ffn": "#ff7f0e",
    "dev_12lead": "#9467bd",
}

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
KEEP_LEAD_IDXS_8 = torch.tensor([0, 1, 6, 7, 8, 9, 10, 11], dtype=torch.long)
LEADS_8 = [LEADS_12[i] for i in KEEP_LEAD_IDXS_8.tolist()]


def find_repo_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    for candidate in [root, *root.parents]:
        if (candidate / "src" / "encoder.py").is_file():
            return candidate
    raise FileNotFoundError("Could not find repository root containing src/encoder.py")


def ensure_repo_on_path(root: Path) -> None:
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)


def read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def best_epoch(metrics_csv: Path, column: str = "erank_ctx_pool") -> int:
    rows = read_metrics(metrics_csv)
    if not rows:
        raise ValueError(f"No rows in {metrics_csv}")
    return int(max(rows, key=lambda row: row[column])["epoch"])


def best_available_checkpoint(
    run_dir: Path,
    metrics_csv: Path,
    column: str = "erank_ctx_pool",
) -> tuple[int, Path, float]:
    """Select the best metric row that has a corresponding saved checkpoint.

    `pretrain.py` logs zero-based metric epochs but writes
    `checkpoint_epoch_{epoch + 1}.pt` after periodic saves. It also writes
    `checkpoint_epoch_0.pt` before training, so that file is intentionally not
    treated as the checkpoint for metric epoch 0.
    """
    rows = read_metrics(metrics_csv)
    candidates: list[tuple[float, int, Path]] = []
    for checkpoint in sorted(run_dir.glob("checkpoint_epoch_*.pt")):
        match = re.fullmatch(r"checkpoint_epoch_(\d+)\.pt", checkpoint.name)
        if not match:
            continue
        saved_epoch = int(match.group(1))
        if saved_epoch == 0:
            continue
        metric_epoch = saved_epoch - 1
        if 0 <= metric_epoch < len(rows):
            candidates.append((float(rows[metric_epoch][column]), metric_epoch, checkpoint))
    for name in ("best_pool_erank.pt", "best_pooled_erank.pt", "checkpoint_best_pool_erank.pt"):
        checkpoint = run_dir / name
        if checkpoint.is_file():
            epoch = best_epoch(metrics_csv, column)
            value = float(rows[epoch][column])
            candidates.append((value, epoch, checkpoint))
    if not candidates:
        raise FileNotFoundError(
            f"No saved checkpoints in {run_dir} could be mapped to rows in {metrics_csv}."
        )
    value, epoch, checkpoint = max(candidates, key=lambda item: item[0])
    return epoch, checkpoint, value


def resolve_checkpoint(run_dir: Path, epoch: int | None = None) -> Path:
    """Find a checkpoint in a run directory without silently choosing a wrong one."""
    extensions = ("*.pt", "*.pth", "*.ckpt")
    all_ckpts = sorted(p for pattern in extensions for p in run_dir.glob(pattern))
    if epoch is not None:
        # `metrics.csv` records zero-based training-loop epochs. `pretrain.py`
        # writes periodic checkpoints after an epoch as checkpoint_epoch_{epoch+1}.
        post_epoch = epoch + 1
        names = [
            f"checkpoint_epoch_{post_epoch}.pt",
            f"checkpoint_epoch_{post_epoch:03d}.pt",
            f"epoch_{post_epoch}.pt",
            f"epoch_{post_epoch:03d}.pt",
            f"checkpoint_epoch_{epoch}.pt",
            f"checkpoint_epoch_{epoch:03d}.pt",
            f"epoch_{epoch}.pt",
            f"epoch_{epoch:03d}.pt",
            f"checkpoint_{epoch}.pt",
            f"checkpoint_{epoch:03d}.pt",
            f"checkpoint_epoch_{epoch}.pth",
            f"checkpoint_epoch_{epoch:03d}.pth",
            f"epoch_{epoch}.pth",
            f"epoch_{epoch:03d}.pth",
            f"checkpoint_epoch_{epoch}.ckpt",
            f"checkpoint_epoch_{epoch:03d}.ckpt",
            "best_pool_erank.pt",
            "best_pooled_erank.pt",
            "checkpoint_best_pool_erank.pt",
        ]
        for name in names:
            candidate = run_dir / name
            if candidate.is_file():
                return candidate
        fuzzy = [
            p for p in all_ckpts
            if f"epoch_{epoch}" in p.stem
            or f"epoch{epoch}" in p.stem
            or f"_{epoch:03d}" in p.stem
            or f"epoch_{post_epoch}" in p.stem
            or f"epoch{post_epoch}" in p.stem
            or f"_{post_epoch:03d}" in p.stem
        ]
        if len(fuzzy) == 1:
            return fuzzy[0]
        if len(fuzzy) > 1:
            raise FileNotFoundError(
                f"Multiple possible checkpoints for epoch {epoch} in {run_dir}: "
                + ", ".join(p.name for p in fuzzy)
            )
    if len(all_ckpts) == 1:
        return all_ckpts[0]
    tried = ", ".join([
        f"checkpoint_epoch_{epoch}.pt" if epoch is not None else "checkpoint_epoch_<epoch>.pt",
        f"checkpoint_epoch_{epoch:03d}.pt" if epoch is not None else "checkpoint_epoch_<epoch:03d>.pt",
        "epoch_<epoch>.pt",
        "best_pool_erank.pt",
        "*.pt/*.pth/*.ckpt when unambiguous",
    ])
    found = ", ".join(p.name for p in all_ckpts) or "none"
    raise FileNotFoundError(
        f"Could not resolve checkpoint in {run_dir} for epoch {epoch}. "
        f"Tried: {tried}. Found checkpoint-like files: {found}."
    )


def unwrap_checkpoint(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("context_encoder", "student", "encoder", "model", "state_dict", "net"):
            inner = obj.get(key)
            if isinstance(inner, dict) and inner and isinstance(next(iter(inner.values())), torch.Tensor):
                return inner
        if obj and isinstance(next(iter(obj.values())), torch.Tensor):
            return obj
    raise TypeError(f"Unexpected checkpoint type: {type(obj)}")


def strip_known_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("context_encoder.", "module.context_encoder.", "student.", "encoder.", "module.")
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        out[new_key] = value
    return out


def build_ours_encoder(run_name: str, root: Path, *, use_flash: bool = True):
    ensure_repo_on_path(root)
    from src.configs.presets import dev_preset, final_preset
    from src.encoder import ECGEncoder
    from src.tokenizer import ECGTokenizer

    if run_name == "final_preset_final":
        tok_cfg, enc_cfg, _ = final_preset()
    else:
        tok_cfg, enc_cfg, _ = dev_preset()

    if run_name in {"dev_100hz", "dev_100hz_ffn"}:
        enc_cfg.num_patches = 20
    if run_name == "dev_100hz_ffn":
        tok_cfg.kind = "ffn"
    elif run_name == "dev_12lead":
        enc_cfg.num_leads = 12

    enc_cfg.use_flash = use_flash
    tokenizer = ECGTokenizer(tok_cfg)
    encoder = ECGEncoder(enc_cfg, tokenizer)
    return encoder, tokenizer, enc_cfg


def load_encoder_weights(encoder: torch.nn.Module, checkpoint: Path, device: str | torch.device = "cpu") -> None:
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    state = strip_known_prefixes(unwrap_checkpoint(blob))
    encoder.load_state_dict(state, strict=True)


def prepare_signal(x: torch.Tensor, variant: str) -> torch.Tensor:
    """Convert PTB-XL `(B, 12, T)` waves to the geometry expected by a run."""
    if variant == "dev_12lead":
        y = x
        target_t = 2500
    else:
        y = x.index_select(dim=1, index=KEEP_LEAD_IDXS_8.to(x.device))
        target_t = 1000 if variant in {"dev_100hz", "dev_100hz_ffn"} else 2500
    if y.shape[-1] != target_t:
        y = F.interpolate(y, size=target_t, mode="linear", align_corners=False)
    return y


def pooled_features(encoder: torch.nn.Module, tokenizer, waves: torch.Tensor) -> torch.Tensor:
    patches = tokenizer.patchify(waves)
    tokens = encoder.forward_all(patches)
    return tokens.mean(dim=1)


def attention_to_matrix(layer_attn: torch.Tensor, sample: int = 0) -> np.ndarray:
    """Return head-averaged `(S, S)` attention for one sample."""
    if layer_attn.ndim != 4:
        raise ValueError(f"Expected attention shape (B, H, S, S), got {tuple(layer_attn.shape)}")
    return layer_attn[sample].mean(dim=0).detach().cpu().numpy()


def lead_boundaries(num_leads: int, num_patches: int) -> tuple[np.ndarray, np.ndarray]:
    boundaries = np.arange(num_patches, num_leads * num_patches, num_patches)
    centers = np.arange(num_patches / 2, num_leads * num_patches, num_patches)
    return boundaries, centers


@contextmanager
def official_attention_hooks(encoder: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    captured: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().cpu())

    for block in encoder.encoder_blocks.blocks:
        handles.append(block.attn.attn_drop.register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
