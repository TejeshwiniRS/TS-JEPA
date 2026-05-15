"""Extract encoder attention maps for TS-JEPA / ECG-JEPA comparison."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


LEAD_ORDER_8 = np.array(["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract encoder attention maps.")
    p.add_argument("--model", choices=["official", "ours"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input_npy", type=Path, default=Path("data/pretrain/X_pretrain_val.npy"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--preset", choices=["dev", "final"], default="final", help="Only used for --model ours.")
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--average_heads",
        action="store_true",
        help="Save (B, S, S) per layer instead of full (B, H, S, S).",
    )
    return p


def load_input_batch(path: Path, start: int, num_samples: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected input npy with shape (N, C, T), got {arr.shape}")
    stop = min(start + num_samples, arr.shape[0])
    if start < 0 or start >= stop:
        raise ValueError(f"Invalid sample window start={start}, stop={stop}, total={arr.shape[0]}")
    batch = torch.from_numpy(np.array(arr[start:stop], dtype=np.float32, copy=True)).to(device)
    indices = np.arange(start, stop, dtype=np.int64)
    return batch, indices


@contextmanager
def official_attention_hooks(encoder: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Capture official ECG-JEPA attention via hooks.

    The official Attention module returns only the projected values, but its
    dropout submodule receives the post-softmax attention tensor. In eval mode
    dropout is identity, so the hook sees the probabilities we want.
    """
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


def load_official_encoder(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    repo = Path(__file__).resolve().parents[1] / "ecg_jepa"
    sys.path.insert(0, str(repo))
    try:
        from models import load_encoder  # type: ignore

        encoder, _embed_dim = load_encoder(str(checkpoint))
    finally:
        if sys.path and sys.path[0] == str(repo):
            sys.path.pop(0)
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def extract_official(checkpoint: Path, signals: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    if signals.shape[1:] != (8, 2500):
        raise ValueError(f"Official ECG-JEPA expects (B, 8, 2500), got {tuple(signals.shape)}")
    encoder = load_official_encoder(checkpoint, device)
    with torch.no_grad(), official_attention_hooks(encoder) as captured:
        _ = encoder.representation(signals)
    return captured


def load_ours_encoder(checkpoint: Path, preset: str, device: torch.device):
    from src.configs import dev_preset, final_preset
    from src.encoder import ECGEncoder
    from src.tokenizer import ECGTokenizer

    tok_cfg, enc_cfg, _pred_cfg = {"dev": dev_preset, "final": final_preset}[preset]()
    enc_cfg.use_flash = False
    tokenizer = ECGTokenizer(tok_cfg)
    encoder = ECGEncoder(enc_cfg, tokenizer)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("context_encoder") or ckpt.get("encoder") or ckpt
    encoder.load_state_dict(state)
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder, tokenizer


def extract_ours(checkpoint: Path, preset: str, signals: torch.Tensor, device: torch.device) -> list[torch.Tensor]:
    encoder, tokenizer = load_ours_encoder(checkpoint, preset, device)
    patches = tokenizer.patchify(signals)
    with torch.no_grad():
        _tokens, attn = encoder.forward_all(patches, return_attn=True)
    return [x.detach().cpu() for x in attn]


def maybe_average_heads(attn: list[torch.Tensor], average_heads: bool) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for layer_attn in attn:
        if average_heads:
            layer_attn = layer_attn.mean(dim=1)
        arrays.append(layer_attn.numpy().astype(np.float32, copy=False))
    return arrays


def save_npz(
    path: Path,
    attn_arrays: list[np.ndarray],
    input_indices: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"layer_{i:02d}": arr
        for i, arr in enumerate(attn_arrays)
    }
    payload["input_indices"] = input_indices
    payload["lead_order"] = LEAD_ORDER_8
    payload["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    signals, indices = load_input_batch(args.input_npy, args.start, args.num_samples, device)

    if args.model == "official":
        attn = extract_official(args.checkpoint, signals, device)
    else:
        attn = extract_ours(args.checkpoint, args.preset, signals, device)

    arrays = maybe_average_heads(attn, args.average_heads)
    metadata = {
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "input_npy": str(args.input_npy),
        "preset": args.preset if args.model == "ours" else None,
        "average_heads": args.average_heads,
        "num_layers": len(arrays),
        "array_shapes": [list(arr.shape) for arr in arrays],
    }
    save_npz(args.out, arrays, indices, metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
