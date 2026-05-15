# Attention comparison

Utilities for extracting and comparing attention maps from:

- our TS-JEPA checkpoints (`src.encoder.ECGEncoder` with `return_attn=True`), and
- the public ECG-JEPA checkpoint in `ecg_jepa/multiblock_epoch100.pth`.

The official ECG-JEPA implementation does not return attention maps, but it
computes them explicitly in `ecg_jepa/ecg_jepa.py::Attention.forward`. We avoid
editing the official repo by registering forward hooks on each block's
`attn.attn_drop` module. In eval mode dropout is inactive, so the hook captures
the post-softmax attention probabilities.

Example:

```bash
# Public ECG-JEPA checkpoint; use the .jepa env because it has timm installed.
.jepa/bin/python -m attention_comparison.extract_attention \
  --model official \
  --checkpoint ecg_jepa/multiblock_epoch100.pth \
  --input_npy data/pretrain/X_pretrain_val.npy \
  --num_samples 2 \
  --out attention_comparison/results/official_val2.npz

# Our final preset checkpoint.
.jepa/bin/python -m attention_comparison.extract_attention \
  --model ours \
  --preset final \
  --checkpoint checkpoints/final_mb/checkpoint_epoch_25.pt \
  --input_npy data/pretrain/X_pretrain_val.npy \
  --num_samples 2 \
  --out attention_comparison/results/ours_final_epoch25_val2.npz
```

Saved `.npz` files contain:

- `layer_00`, `layer_01`, ...: attention tensors with shape `(B, H, S, S)`.
- `input_indices`: the selected sample indices from the input `.npy`.
- `lead_order`: lead names for interpreting the sequence layout.

For both implementations the sequence is lead-major:

```text
I[0:50], II[0:50], V1[0:50], ..., V6[0:50]
```

The official model uses CroPA/cross-pattern masking, so many attention entries
are structurally zero. Our model uses full self-attention.

## Summarizing extracted maps

Once an `.npz` is saved, `summarize_attention.py` turns it into heatmaps,
profiles, and a small CSV that can be diffed across models.

```bash
.jepa/bin/python -m attention_comparison.summarize_attention \
  attention_comparison/results/smoke_official_val1_avgheads.npz \
  --layers 0 5 11
```

Arguments:

- positional: path to the `.npz` produced by `extract_attention.py`.
- `--out_dir`: where to write outputs. Defaults to the `.npz` path with the
  `.npz` suffix stripped.
- `--sample`: index inside the batch axis `B` to summarize. Default `0`.
- `--layers`: indices of encoder layers to summarize. Omit to process all
  layers.
- `--num_leads` / `--num_patches`: must match the token grid used at
  extraction time. Default `8` and `50` for the ECG-JEPA setup.

For each selected layer the script writes:

- `layer_<L>_patch_matrix.png`: the full token-by-token attention heatmap.
  The matrix is `(S, S)`, where `S = num_leads * num_patches`; rows are query
  tokens and columns are key tokens. Since tokens are lead-major, every
  `num_patches` rows/columns correspond to one lead. White grid lines mark
  lead boundaries, so each large block is a query-lead/key-lead pair and each
  pixel inside a block is patch-to-patch attention.
- `layer_<L>_lead_matrix.png`: an `(num_leads, num_leads)` heatmap of average
  attention from every query lead to every key lead. To build it, the
  `(S, S)` head-averaged matrix is reshaped into
  `(num_leads, num_patches, num_leads, num_patches)` and averaged over the
  two patch axes, leaving `lead_attention[q_lead, k_lead]`. Diagonal entries
  show how much each lead attends within itself; off-diagonal entries show
  cross-lead routing.
- `layer_<L>_temporal_profile.png`: mean attention as a function of
  `|q_patch - k_patch|`. For every query/key patch pair we add their head-
  averaged attention to a bucket indexed by their absolute patch distance,
  then divide by the number of pairs in that bucket. Distance `0` is
  same-patch attention; larger distances probe whether the layer keeps
  long-range temporal context.

The script also writes:

- `summary.csv`: one row per selected layer with the diagonal/off-diagonal
  lead-attention means and the same-/near-/far-patch entries from the
  temporal profile.
- `metadata.json`: a copy of the metadata that was embedded in the `.npz`
  by `extract_attention.py`.

Caveat for the official model: CroPA masks attention so each token can only
attend to same-lead temporal tokens and same-time cross-lead tokens. The
average heatmaps therefore reflect attention mass over the allowed
connections only, and are not directly comparable in absolute terms to our
full-self-attention model. Comparing patterns (which leads or distances
receive more mass relative to what is allowed) is the intended use.
