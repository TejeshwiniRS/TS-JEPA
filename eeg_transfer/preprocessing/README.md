# BCIC-IV 2a preprocessing

Single-script pipeline that turns the BCIC-IV 2a `.mat` files into three
`.npz` splits (train / val / test) ready to feed into the fine-tuning
notebooks under `eeg_transfer/notebooks/`.

## Run

From the repo root:

```bash
/home/aimakeradmin/shady-env/bin/python3 eeg_transfer/preprocessing/preprocess_bcic2a.py
```

Outputs (under `eeg_transfer/data/processed/`):

| file | content |
|---|---|
| `train.npz` | `X: (N_train, 22, 1000) float32`, `y: (N_train,) int64` |
| `val.npz`   | same, val subjects |
| `test.npz`  | same, test subjects |
| `norm_stats.npz` | per-channel `mu`, `sigma` fit on train |
| `meta.json` | label map, splits, filter params, trial counts |

## Protocol — follows EEG-FM-Bench (arXiv 2508.17742, Appendix B)

| step | choice |
|---|---|
| Channel selection | first 22 of 25 columns = EEG; last 3 EOG dropped |
| Bandpass filter | 0.1–100 Hz Butterworth (order 4), zero-phase (`sosfiltfilt`) |
| Notch filter | 50 Hz (BCIC-IV 2a is European data → 50 Hz mains) |
| Resampling | none — source is 250 Hz, our model patch=50 → 4 s = 1000 samples = 20 patches |
| Trial window | `[t_cue + 0.5 s, t_cue + 4.5 s]` = `[trial + 625, trial + 1625]` (1000 samples). Avoids the cue-onset visual evoked response and covers the canonical motor-imagery period (3–6 s). |
| Subject split | train = subjects 1–5, val = 6–7, test = 8–9 (matches EEG-FM-Bench Appendix B.3) |
| Artifact rejection | drop trials whose `artifacts` flag in the `.mat` is set |
| Normalization | per-channel z-score, μ/σ fit on train only, applied to all splits |
| Label remap | original labels {1,2,3,4} → {0,1,2,3} for `nn.CrossEntropyLoss` (left=0, right=1, feet=2, tongue=3) |

## Limitation

Only the **T (training-session)** files (`A0[1-9]T.mat`) are present in this
repo. EEG-FM-Bench Table 1 reports 2784 / 1152 / 1152 trials, implying both
T and E sessions were used. Our trial counts are about half of those:
expect roughly **~1440 train / ~576 val / ~576 test** before artifact
rejection. Numbers remain comparable in interpretation; only the absolute
training-data scale is smaller.

## Quick verification

```python
import numpy as np
d = np.load("eeg_transfer/data/processed/train.npz")
print(d["X"].shape, d["X"].dtype, d["y"].shape, np.bincount(d["y"]))
```

Expected: `(~1400, 22, 1000) float32, (~1400,) int64`, roughly balanced 4-class distribution.
