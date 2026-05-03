# Pretraining data pipeline (Chapman + Ningbo + CODE-15)

Builds the unified pretraining arrays consumed by `src.data.PretrainECGDataset`:

```
<out_dir>/
  X_pretrain_train.npy   # (N_train, 8, 2500) float32
  X_pretrain_val.npy     # (N_val,   8, 2500) float32
  processed_pretrain.csv # one row per accepted record
  failed_pretrain.csv    # one row per rejected record + reason
  run_config.json
  run_summary.json
```

Per the ECG-JEPA paper (Kim 2026, Section 4):
- 8 leads (`I, II, V1-V6`); the 4 derivable leads are dropped via Einthoven's law.
- 10 s @ 250 Hz -> `T = 2500` samples per record.
- Raw signals — **no normalization, no filtering**.
- "Drop records with missing values" -> any NaN/Inf or any all-zero lead is rejected.

## Source layouts

| Dataset | Hosting | Format | Sample rate | Notes |
| --- | --- | --- | --- | --- |
| Chapman + Ningbo (Zheng 2022, "ecg-arrhythmia 1.0.0") | PhysioNet | WFDB (`.hea` + `.mat`) | 500 Hz | 45,152 records; streamed via `wfdb.rdrecord(pn_dir=...)` |
| CODE-15% (Ribeiro 2021, Zenodo `4916206`) | Zenodo | 18 zipped HDF5 shards | 400 Hz | ~46 GB total; only 10 s records kept (paper) |

CODE-15 records are zero-padded to 4096 samples; the loader trims leading/trailing zero rows and only keeps records whose core (non-padded) length is at least 4000 samples (= 10 s).

## Dependencies

```bash
pip install numpy scipy h5py wfdb
```

`requests` is not required; the CODE-15 downloader uses `urllib.request`.

## Usage

```bash
# Smoke test (1 CODE-15 shard, 50 records):
python -m scripts.pretrain_data.build_pretrain_npy \
    --datasets code15 \
    --code15_shards 0 --code15_max_per_shard 50 \
    --out_dir data/pretrain_smoke

# Chapman + Ningbo only (~43k records, no big download since wfdb streams):
python -m scripts.pretrain_data.build_pretrain_npy \
    --datasets chapman_ningbo \
    --out_dir data/pretrain

# Full paper-faithful corpus (Chapman + Ningbo + CODE-15, ~170k records):
python -m scripts.pretrain_data.build_pretrain_npy \
    --datasets chapman_ningbo code15 \
    --code15_cache /scratch/$USER/code15 \
    --out_dir data/pretrain
```

On MSI, point `--code15_cache` at `/scratch/$USER/...` since each shard is ~5 GB unzipped and you don't want the home filesystem to fill up. The downloader processes shards one at a time and (by default) deletes the zip + HDF5 after each shard; pass `--code15_keep_downloads` to retain them.

## Resuming

Pass `--resume` to skip records whose ids are already in `processed_pretrain.csv`. The chunk writer also picks up where it left off (chunks are atomic), so an interrupted run can be safely restarted with the same command + `--resume`.

## Splits

Splits are decided per record via a deterministic hash of `record_id` modulo 1e6, compared against `--val_ratio` (default 5%). This avoids needing a global manifest, makes the train/val assignment stable across reruns, and means new datasets can be appended later without re-shuffling existing records.
