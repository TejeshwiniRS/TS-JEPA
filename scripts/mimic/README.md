# MIMIC-IV-ECG → NPY pretraining pipeline

A standalone CLI that turns a `(subject_id, study_id)` manifest into the
`(N, 12, 1000)` float32 NPY files consumed by JEPA pretraining. It is kept
isolated from `src/` so that MIMIC-specific concerns (PhysioNet access,
WFDB decoding, resampling, cohort splits) never leak into model code.

## What it produces

Inside `--out_dir` (default `/Users/shadyali/TS-JEPA/data/mimic`):

| File | Shape / content |
| --- | --- |
| `X_ecg_train.npy` | `(N_train, 12, 1000)` float32, raw (no normalization) |
| `X_ecg_val.npy`   | `(N_val, 12, 1000)` float32, raw (no normalization) |
| `processed_records.csv` | append-only log of successful `(subject_id, study_id, split)` |
| `failed_records.csv`    | append-only log of failures with error message |
| `run_config.json`       | snapshot of the CLI config for reproducibility |
| `run_summary.json`      | final counts, paths, and elapsed time |

No labels (`y_*`) and no `norm_*` files are written — this pipeline is
pretraining-only and uses raw fp32 signals.

## Signal processing

1. Stream the `.hea` + `.dat` for each record directly from PhysioNet using
   `wfdb.rdrecord(..., pn_dir=...)`. MIMIC-IV-ECG v1.0 is open access so no
   credentials are required.
2. Reshape to `(12, 5000)` float32 (10 s @ 500 Hz).
3. Replace any non-finite samples (NaN/Inf in unusable leads) with 0.
4. Resample to `(12, 1000)` with `scipy.signal.resample`, which uses
   Fourier-method resampling.

## CLI

```bash
python -m scripts.mimic.build_mimic_npy \
    --ids_csv  /Users/shadyali/TS-JEPA/mimic_ids.csv \
    --out_dir  /Users/shadyali/TS-JEPA/data/mimic \
    --num_workers 16 \
    --max_records 500          # omit for full run (~300k records)
```

Important flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--ids_csv` | `mimic_ids.csv` at repo root | manifest with `subject_id,study_id` |
| `--out_dir` | `data/mimic` | where to write the final NPYs and logs |
| `--num_workers` | `cpu_count() - 1` | parallel download+decode workers |
| `--max_records` | all | cap for smoke tests / partial runs |
| `--val_ratio` | `0.10` | subject-level holdout fraction |
| `--seed` | `0` | deterministic split / subsampling |
| `--chunk_size` | `2000` | records buffered in RAM per split before flushing to a temp chunk file |
| `--retries` | `3` | per-record network retry count |
| `--resume` | off | skip records already listed in `processed_records.csv` |

## Splits

- Subject-level split — no subject appears in both train and val.
- Deterministic for a given `--seed`, so re-running with the same seed
  reproduces the exact same split (useful for resumes).

## Crash-safety / resume

- Each successful record is immediately logged to `processed_records.csv`.
- Records are flushed to sharded `chunk_XXXXX.npy` files under
  `<out_dir>/_tmp/<split>/` every `--chunk_size` records.
- On `--resume`, any `(subject_id, study_id)` already in
  `processed_records.csv` is skipped.
- Final `X_ecg_{train,val}.npy` is assembled from the chunk files via a
  memory-mapped write (no need to fit the whole dataset in RAM).
- The `_tmp/` directory is removed only after the final NPY is written.

## Cloud execution tips

The full 300k-record run is network-bound. The pipeline is designed so that
you can:

- Run the script directly on the cloud VM that will do training — no need to
  download the 90 GB MIMIC archive locally first.
- Use many workers (`--num_workers 32` or `64`) on a reasonable VM; each
  worker is doing HTTP + light CPU resampling, so higher concurrency helps.
- Kill and resume with `--resume` at any time; only in-flight records in the
  current chunk buffer may be lost.

## Dependencies

- `numpy`
- `scipy`
- `wfdb` (PhysioNet WFDB reader)

Install (example):

```bash
pip install numpy scipy wfdb
```
