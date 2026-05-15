"""Summarize extracted attention maps into comparison-friendly plots/CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEAD_ORDER = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize attention npz files.")
    p.add_argument("npz", type=Path)
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--num_leads", type=int, default=8)
    p.add_argument("--num_patches", type=int, default=50)
    return p


def load_layers(path: Path) -> tuple[list[np.ndarray], dict]:
    data = np.load(path, allow_pickle=False)
    layer_keys = sorted(k for k in data.files if k.startswith("layer_"))
    layers = [data[k] for k in layer_keys]
    metadata = {}
    if "metadata_json" in data.files:
        metadata = json.loads(str(data["metadata_json"]))
    return layers, metadata


def ensure_bhss(layer: np.ndarray) -> np.ndarray:
    """Normalize layer array to (B, H, S, S)."""
    if layer.ndim == 3:
        return layer[:, None, :, :]
    if layer.ndim == 4:
        return layer
    raise ValueError(f"Expected layer attention ndim 3 or 4, got shape {layer.shape}")


def lead_matrix(attn_ss: np.ndarray, num_leads: int, num_patches: int) -> np.ndarray:
    grid = attn_ss.reshape(num_leads, num_patches, num_leads, num_patches)
    return grid.mean(axis=(1, 3))


def temporal_profile(attn_ss: np.ndarray, num_leads: int, num_patches: int) -> np.ndarray:
    grid = attn_ss.reshape(num_leads, num_patches, num_leads, num_patches)
    max_delta = num_patches - 1
    sums = np.zeros(max_delta + 1, dtype=np.float64)
    counts = np.zeros(max_delta + 1, dtype=np.float64)
    for q in range(num_patches):
        for k in range(num_patches):
            d = abs(q - k)
            sums[d] += grid[:, q, :, k].mean()
            counts[d] += 1
    return sums / np.maximum(counts, 1)


def save_heatmap(matrix: np.ndarray, path: Path, title: str, labels: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_patch_heatmap(
    attn_ss: np.ndarray,
    path: Path,
    title: str,
    num_leads: int,
    num_patches: int,
) -> None:
    """Save the full token-by-token attention map with lead boundaries.

    The axes are still token indices, but because tokens are ordered
    lead-major, every `num_patches` rows/columns form one lead block.
    """
    labels = LEAD_ORDER[:num_leads]
    boundaries = np.arange(num_patches, num_leads * num_patches, num_patches)
    centers = np.arange(num_patches / 2, num_leads * num_patches, num_patches)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_ss, cmap="viridis", aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("key token: lead-major patch index")
    ax.set_ylabel("query token: lead-major patch index")
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.8)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.5, alpha=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_profile(profile: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(profile.shape[0]), profile)
    ax.set_xlabel("absolute temporal patch distance")
    ax.set_ylabel("mean attention")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir or args.npz.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    layers, metadata = load_layers(args.npz)
    selected = args.layers if args.layers is not None else list(range(len(layers)))

    rows = []
    for layer_idx in selected:
        layer = ensure_bhss(layers[layer_idx])
        if args.sample >= layer.shape[0]:
            raise ValueError(f"sample={args.sample} out of range for layer shape {layer.shape}")
        attn_ss = layer[args.sample].mean(axis=0)

        lead_attn = lead_matrix(attn_ss, args.num_leads, args.num_patches)
        profile = temporal_profile(attn_ss, args.num_leads, args.num_patches)

        save_patch_heatmap(
            attn_ss,
            out_dir / f"layer_{layer_idx:02d}_patch_matrix.png",
            f"Layer {layer_idx:02d}: patch-level attention",
            args.num_leads,
            args.num_patches,
        )
        save_heatmap(
            lead_attn,
            out_dir / f"layer_{layer_idx:02d}_lead_matrix.png",
            f"Layer {layer_idx:02d}: mean attention by lead",
            labels=LEAD_ORDER[: args.num_leads],
        )
        save_profile(
            profile,
            out_dir / f"layer_{layer_idx:02d}_temporal_profile.png",
            f"Layer {layer_idx:02d}: attention by temporal distance",
        )

        rows.append({
            "layer": layer_idx,
            "diag_lead_attention": float(np.trace(lead_attn) / args.num_leads),
            "offdiag_lead_attention": float((lead_attn.sum() - np.trace(lead_attn)) / (args.num_leads * (args.num_leads - 1))),
            "same_patch_attention": float(profile[0]),
            "near_patch_attention_d1": float(profile[1]) if profile.shape[0] > 1 else float("nan"),
            "far_patch_attention_last": float(profile[-1]),
        })

    with (out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
