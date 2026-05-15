"""Preprocess BCIC-IV 2a .mat files into train/val/test .npz splits.

Protocol follows EEG-FM-Bench (arXiv 2508.17742, Appendix B):
  - Channel selection: 22 EEG channels (drop 3 EOG channels at the end).
  - Filtering: band-pass 0.1-100 Hz (Butterworth, filtfilt) + notch 50 Hz.
  - Resampling: none. Source = 250 Hz; our model patch=50 → 4s = 1000 samples = 20 patches.
  - Segmentation: extract 4-second motor-imagery windows.
      Per BCIC-IV 2a protocol: trial starts at t=0, cue at t=2s, MI from t=3s to t=6s.
      We extract the canonical window [t_cue + 0.5s, t_cue + 4.5s] = [trial+625, trial+1625]
      (avoids the cue-onset visual evoked response).
  - Subject split (matches EEG-FM-Bench Appendix B.3):
      train = subjects 1-5, val = 6-7, test = 8-9.
  - Artifact rejection: drop trials whose `artifacts` flag is set.
  - Per-channel z-score: μ, σ fit on training set only, applied to all splits.

Only T (training-session) files are present in this repo. EEG-FM-Bench used T+E,
so our trial counts are about half theirs (~1440 train, ~576 val, ~576 test).

Usage:
    python eeg_transfer/preprocessing/preprocess_bcic2a.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos

# --- Constants -----------------------------------------------------------------
FS = 250  # Hz, source sampling rate of BCIC-IV 2a
N_CHANNELS_EEG = 22  # first 22 of 25 are EEG; last 3 are EOG
WINDOW_LEN_SAMPLES = 1000  # 4 seconds at 250 Hz
CUE_OFFSET_SAMPLES = 500  # cue appears at t=2s after trial start
WINDOW_START_OFFSET_FROM_CUE = 125  # 0.5 s after cue → trial + 625
WINDOW_START_FROM_TRIAL = CUE_OFFSET_SAMPLES + WINDOW_START_OFFSET_FROM_CUE  # 625

NOTCH_HZ = 50.0  # European mains
BANDPASS_LOW = 0.1
BANDPASS_HIGH = 100.0  # well below Nyquist (125 Hz)

SUBJECT_SPLITS = {
    "train": [1, 2, 3, 4, 5],
    "val":   [6, 7],
    "test":  [8, 9],
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # .../TS-JEPA/
RAW_DIR = REPO_ROOT / "eeg_transfer" / "data" / "bcic-2a"
OUT_DIR = REPO_ROOT / "eeg_transfer" / "data" / "processed"


# --- Filtering -----------------------------------------------------------------
def _build_filters(fs: int):
    """Return (bandpass_sos, notch_sos) ready for sosfiltfilt."""
    nyq = fs / 2.0
    bp_sos = butter(N=4, Wn=[BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq],
                    btype="bandpass", output="sos")
    # iirnotch returns (b, a); convert to sos for filtfilt
    b_n, a_n = iirnotch(w0=NOTCH_HZ / nyq, Q=30.0)
    notch_sos = tf2sos(b_n, a_n)
    return bp_sos, notch_sos


def _apply_filters(x: np.ndarray, bp_sos, notch_sos) -> np.ndarray:
    """Apply bandpass + notch along the time axis. x: (T, C) float."""
    # sosfiltfilt operates on the last axis by default
    y = sosfiltfilt(bp_sos, x, axis=0)
    y = sosfiltfilt(notch_sos, y, axis=0)
    return y


# --- Per-subject extraction ----------------------------------------------------
def load_subject(subj_id: int, bp_sos, notch_sos) -> tuple[np.ndarray, np.ndarray]:
    """Extract all motor-imagery trials for one subject.

    Returns:
        X: (n_trials, 22, 1000) float32 — channels × samples per trial
        y: (n_trials,) int64 — labels in {0, 1, 2, 3}
    """
    path = RAW_DIR / f"A0{subj_id}T.mat"
    if not path.exists():
        raise FileNotFoundError(path)

    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    data = mat["data"]  # length-9 array of structs

    X_list, y_list = [], []
    for run in data:
        # Calibration runs have empty trial/y arrays — skip them.
        trials = run.trial
        labels = run.y
        artifacts = run.artifacts
        if not hasattr(trials, "shape") or trials.size == 0:
            continue

        # Filter the run's continuous signal once, then slice trials out.
        X_run = np.asarray(run.X, dtype=np.float64)  # (T, 25)
        # Drop EOG channels (last 3) before filtering to save work.
        X_run = X_run[:, :N_CHANNELS_EEG]  # (T, 22)
        X_run = _apply_filters(X_run, bp_sos, notch_sos)

        for t_idx, t_start, lbl, art in zip(range(len(trials)), trials, labels, artifacts):
            if int(art) != 0:
                continue  # skip artifact-flagged trials
            start = int(t_start) + WINDOW_START_FROM_TRIAL
            end = start + WINDOW_LEN_SAMPLES
            if end > X_run.shape[0]:
                # Defensive: trial too close to end of run. Should not happen
                # for valid BCIC-IV 2a recordings, but guard anyway.
                continue
            seg = X_run[start:end, :]  # (1000, 22)
            X_list.append(seg.T.astype(np.float32))  # → (22, 1000)
            y_list.append(int(lbl) - 1)  # 1..4 → 0..3

    if not X_list:
        raise RuntimeError(f"No usable trials found for subject {subj_id}")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# --- Pipeline ------------------------------------------------------------------
def main() -> None:
    if not RAW_DIR.exists():
        sys.exit(f"Raw data dir not found: {RAW_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bp_sos, notch_sos = _build_filters(FS)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name, subj_ids in SUBJECT_SPLITS.items():
        Xs, ys = [], []
        for sid in subj_ids:
            print(f"[{split_name}] loading subject {sid} ...", flush=True)
            X, y = load_subject(sid, bp_sos, notch_sos)
            print(f"          → X={X.shape}, label dist={np.bincount(y, minlength=4).tolist()}")
            Xs.append(X)
            ys.append(y)
        X_split = np.concatenate(Xs, axis=0)
        y_split = np.concatenate(ys, axis=0)
        splits[split_name] = (X_split, y_split)
        print(f"[{split_name}] total trials={X_split.shape[0]}")

    # Per-channel z-score: fit on train, apply to all.
    X_train = splits["train"][0]
    # mean over (trials, time) for each channel → shape (22,)
    mu = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 22, 1)
    sigma = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    print(f"\nNormalization (train-only fit): "
          f"per-channel μ range [{mu.min():.4f}, {mu.max():.4f}], "
          f"σ range [{sigma.min():.4f}, {sigma.max():.4f}]")

    for split_name, (X, y) in splits.items():
        X_norm = ((X - mu) / sigma).astype(np.float32)
        out_path = OUT_DIR / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X_norm, y=y)
        print(f"wrote {out_path}  shape X={X_norm.shape}, y={y.shape}")

    # Save normalization stats and split metadata for reproducibility.
    stats_path = OUT_DIR / "norm_stats.npz"
    np.savez(stats_path, mu=mu.squeeze(), sigma=sigma.squeeze())
    print(f"wrote {stats_path}")

    meta = {
        "fs": FS,
        "n_channels": N_CHANNELS_EEG,
        "window_samples": WINDOW_LEN_SAMPLES,
        "window_start_from_trial_samples": WINDOW_START_FROM_TRIAL,
        "bandpass_hz": [BANDPASS_LOW, BANDPASS_HIGH],
        "notch_hz": NOTCH_HZ,
        "subject_splits": SUBJECT_SPLITS,
        "label_map": {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3},
        "trial_counts": {k: int(v[1].shape[0]) for k, v in splits.items()},
        "label_dist": {k: np.bincount(v[1], minlength=4).tolist()
                       for k, v in splits.items()},
        "source_files": "A0[1-9]T.mat (training sessions only; E sessions not used)",
    }
    meta_path = OUT_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
