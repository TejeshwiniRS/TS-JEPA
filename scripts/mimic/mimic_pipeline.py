"""Core logic for the MIMIC-IV-ECG to NPY pretraining pipeline.

This module is intentionally decoupled from the CLI wrapper in `build_mimic_npy.py`
so that the individual stages (manifest, split, worker, finalize) can be tested or
reused without going through argparse.

Pipeline overview:
  1. Load `(subject_id, study_id)` pairs from the ids CSV.
  2. Deterministic subject-level split into train / val.
  3. Worker pool downloads each record from PhysioNet (open access),
     decodes it with WFDB, resamples from 500 Hz (5000 samples) to 100 Hz
     (1000 samples) per lead, and returns a `(12, 1000)` float32 array.
  4. A main-process writer accumulates records into per-split chunk files on
     disk, so the run is crash-safe / resumable and never needs to hold the
     full dataset in RAM.
  5. On success, chunk files are concatenated into `X_ecg_train.npy` and
     `X_ecg_val.npy` inside the output directory.

The MIMIC-IV-ECG layout on PhysioNet:
    files/pNNNN/pXXXXXXXX/sZZZZZZZZ/ZZZZZZZZ.{hea,dat}
where NNNN is the first 4 characters of subject_id (zero-padded to 8 digits)
and ZZZZZZZZ is the study id (zero-padded to 8 digits).
"""

from __future__ import annotations

import csv
import json
import logging
import re
import multiprocessing as mp
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

LOG = logging.getLogger("mimic_pipeline")

_CHUNK_NPY_RE = re.compile(r"^chunk_(\d{5})\.npy$")

TARGET_LEADS: int = 12
SOURCE_LENGTH: int = 5000  # 10 s @ 500 Hz
TARGET_LENGTH: int = 1000  # 10 s @ 100 Hz
DEFAULT_BASE_URL: str = "https://physionet.org/files/mimic-iv-ecg/1.0"
# `wfdb.rdrecord(pn_dir=...)` expects the path without the leading
# "https://physionet.org/files/" prefix.
DEFAULT_PN_DIR_ROOT: str = "mimic-iv-ecg/1.0"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    ids_csv: Path
    out_dir: Path
    num_workers: int
    max_records: Optional[int] = None
    val_ratio: float = 0.10
    seed: int = 0
    tmp_dir: Optional[Path] = None
    base_url: str = DEFAULT_BASE_URL
    pn_dir_root: str = DEFAULT_PN_DIR_ROOT
    chunk_size: int = 2000
    resume: bool = False
    retries: int = 3
    retry_backoff_s: float = 2.0

    def to_serializable(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d


# ---------------------------------------------------------------------------
# Manifest: ids CSV -> (subject, study) records + split assignment
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    subject_id: str
    study_id: str
    split: str  # "train" or "val"


def _pad_id(x: str, width: int = 8) -> str:
    x = str(x).strip()
    return x.zfill(width)


def record_pn_dir(subject_id: str, study_id: str, pn_dir_root: str) -> str:
    """Build the `pn_dir` argument passed to `wfdb.rdrecord`.

    Example for subject_id=10001725, study_id=41420867:
        mimic-iv-ecg/1.0/files/p1000/p10001725/s41420867
    """
    sid = _pad_id(subject_id)
    tid = _pad_id(study_id)
    group = f"p{sid[:4]}"
    return f"{pn_dir_root}/files/{group}/p{sid}/s{tid}"


def record_name(study_id: str) -> str:
    return _pad_id(study_id)


def load_ids_csv(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "subject_id" not in reader.fieldnames or "study_id" not in reader.fieldnames:
            raise ValueError(
                f"ids CSV {path} must contain columns 'subject_id' and 'study_id'; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            key = (str(row["subject_id"]).strip(), str(row["study_id"]).strip())
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def subject_level_split(
    pairs: list[tuple[str, str]], val_ratio: float, seed: int
) -> dict[str, str]:
    """Return a mapping subject_id -> "train" | "val" such that no subject is
    present in both splits. Deterministic for a given `seed`."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0,1); got {val_ratio}")
    subjects = sorted({s for s, _ in pairs})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, int(round(len(subjects) * val_ratio)))
    val_set = set(subjects[:n_val])
    return {s: ("val" if s in val_set else "train") for s in subjects}


def build_manifest(
    pairs: list[tuple[str, str]],
    val_ratio: float,
    seed: int,
    max_records: Optional[int],
) -> list[ManifestEntry]:
    if max_records is not None and max_records < len(pairs):
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_records)
    assignment = subject_level_split(pairs, val_ratio=val_ratio, seed=seed)
    return [
        ManifestEntry(subject_id=s, study_id=t, split=assignment[s])
        for (s, t) in pairs
    ]


# ---------------------------------------------------------------------------
# Worker: download + decode + resample
# ---------------------------------------------------------------------------


def _resample_5000_to_1000(signal: np.ndarray) -> np.ndarray:
    """Resample along the time axis (axis=-1) from 5000 to 1000 samples.

    Uses `scipy.signal.resample`, which performs Fourier-method resampling.
    Input shape: ``(..., 5000)`` with time last.
    """
    from scipy.signal import resample

    if signal.shape[-1] != SOURCE_LENGTH:
        raise ValueError(
            f"expected last dim {SOURCE_LENGTH}, got {signal.shape[-1]}"
        )
    out = resample(signal, TARGET_LENGTH, axis=-1)
    if out.shape[-1] != TARGET_LENGTH:
        raise RuntimeError(
            f"resample produced length {out.shape[-1]}, expected {TARGET_LENGTH}"
        )
    return np.asarray(out, dtype=np.float32)


def _validate_runtime_dependencies() -> None:
    """Fail fast in the parent process if worker-time imports are broken."""
    import scipy.signal  # noqa: F401
    import wfdb  # noqa: F401


def _fetch_record_array(
    subject_id: str,
    study_id: str,
    pn_dir_root: str,
    retries: int,
    retry_backoff_s: float,
) -> np.ndarray:
    """Return the raw 12-lead signal as (12, 5000) float32.

    Uses `wfdb.rdrecord` with `pn_dir=` which streams the `.hea` + `.dat`
    directly from PhysioNet.
    """
    import wfdb  # lazy import so argparse --help works without wfdb installed

    pn_dir = record_pn_dir(subject_id, study_id, pn_dir_root)
    name = record_name(study_id)

    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            rec = wfdb.rdrecord(name, pn_dir=pn_dir)
            sig = rec.p_signal  # (T, C) float64
            if sig is None:
                raise RuntimeError("rdrecord returned empty p_signal")
            # Transpose to (C, T) for consistency with the rest of the stack.
            sig = np.asarray(sig, dtype=np.float32).T
            return sig
        except BaseException as e:  # noqa: BLE001 - network-level retry
            last_err = e
            if attempt < retries:
                time.sleep(retry_backoff_s * attempt)
            else:
                raise
    # Unreachable, but keeps type-checkers happy.
    raise RuntimeError(f"failed after {retries} retries: {last_err}")


def _process_single(
    args: tuple[str, str, str, PipelineConfig],
) -> tuple[str, str, str, str, Optional[np.ndarray], Optional[str]]:
    """Worker entry point. Returns (subject_id, study_id, split, status,
    array_or_none, error_or_none)."""
    subject_id, study_id, split, cfg = args
    try:
        sig = _fetch_record_array(
            subject_id=subject_id,
            study_id=study_id,
            pn_dir_root=cfg.pn_dir_root,
            retries=cfg.retries,
            retry_backoff_s=cfg.retry_backoff_s,
        )
        if sig.ndim != 2 or sig.shape[0] != TARGET_LEADS:
            raise ValueError(
                f"unexpected shape {sig.shape}, expected ({TARGET_LEADS},T)"
            )
        if sig.shape[1] != SOURCE_LENGTH:
            raise ValueError(
                f"unexpected source length {sig.shape[1]}, expected {SOURCE_LENGTH}"
            )
        if not np.isfinite(sig).all():
            # MIMIC ECGs occasionally contain NaN/Inf for unusable leads.
            # Replace with zeros rather than dropping the record.
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

        resampled = _resample_5000_to_1000(sig).astype(np.float32, copy=False)
        if resampled.shape != (TARGET_LEADS, TARGET_LENGTH):
            raise ValueError(
                f"resampled shape {resampled.shape} != ({TARGET_LEADS},{TARGET_LENGTH})"
            )
        return (subject_id, study_id, split, "ok", resampled, None)
    except BaseException as e:  # noqa: BLE001 - worker must never raise
        return (subject_id, study_id, split, "fail", None, repr(e))


# ---------------------------------------------------------------------------
# Chunk writer: buffered, crash-safe, resume-friendly
# ---------------------------------------------------------------------------


class ChunkWriter:
    """Buffers worker outputs for one split and flushes them to sharded
    `.npy` chunk files under `<out_dir>/_tmp/<split>/chunk_XXXXX.npy`.

    On `close()` the chunks are concatenated into a single `X_ecg_<split>.npy`
    inside `out_dir`. The temp directory is retained until finalization so
    the run is fully resumable."""

    def __init__(self, split: str, out_dir: Path, chunk_size: int):
        self.split = split
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.tmp_dir = out_dir / "_tmp" / split
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._buf: list[np.ndarray] = []
        self._next_chunk_idx = self._scan_existing_chunks()
        self._count = self._count_from_existing()

    def _scan_existing_chunks(self) -> int:
        """Next chunk index after the highest `chunk_NNNNN.npy` (complete files only)."""
        indices: list[int] = []
        for p in self.tmp_dir.glob("chunk_*.npy"):
            m = _CHUNK_NPY_RE.match(p.name)
            if m:
                indices.append(int(m.group(1)))
        if not indices:
            return 0
        return max(indices) + 1

    def _count_from_existing(self) -> int:
        total = 0
        for p in sorted(self.tmp_dir.glob("chunk_*.npy")):
            if not _CHUNK_NPY_RE.match(p.name):
                continue
            total += int(np.load(p, mmap_mode="r").shape[0])
        return total

    @property
    def count(self) -> int:
        return self._count + len(self._buf)

    def append(self, arr: np.ndarray) -> None:
        self._buf.append(arr)
        if len(self._buf) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        chunk = np.stack(self._buf, axis=0).astype(np.float32, copy=False)
        path = self.tmp_dir / f"chunk_{self._next_chunk_idx:05d}.npy"
        # `np.save` only treats paths ending in `.npy` as final; otherwise it
        # appends `.npy` (so `foo.npy.partial` would become `foo.npy.partial.npy`).
        tmp_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        np.save(tmp_path, chunk)
        os.replace(tmp_path, path)
        self._count += chunk.shape[0]
        self._next_chunk_idx += 1
        self._buf.clear()

    def finalize(self) -> Path:
        """Concatenate chunk files into the final npy and return its path."""
        self.flush()
        chunk_paths = sorted(
            p
            for p in self.tmp_dir.glob("chunk_*.npy")
            if _CHUNK_NPY_RE.match(p.name)
        )
        if not chunk_paths:
            # Materialize an empty well-formed array so downstream code doesn't
            # have to branch on "missing file".
            final = np.empty((0, TARGET_LEADS, TARGET_LENGTH), dtype=np.float32)
            final_path = self.out_dir / f"X_ecg_{self.split}.npy"
            np.save(final_path, final)
            return final_path

        total = 0
        for p in chunk_paths:
            total += int(np.load(p, mmap_mode="r").shape[0])

        final_path = self.out_dir / f"X_ecg_{self.split}.npy"
        tmp_final = final_path.with_suffix(".npy.partial")
        out = np.lib.format.open_memmap(
            tmp_final,
            mode="w+",
            dtype=np.float32,
            shape=(total, TARGET_LEADS, TARGET_LENGTH),
        )
        offset = 0
        for p in chunk_paths:
            arr = np.load(p, mmap_mode="r")
            n = arr.shape[0]
            out[offset : offset + n] = arr
            offset += n
        out.flush()
        del out
        os.replace(tmp_final, final_path)

        # Keep the tmp chunks until the caller (pipeline) decides to delete them;
        # this lets a user re-run finalization without re-downloading.
        return final_path

    def cleanup_tmp(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)


# ---------------------------------------------------------------------------
# Logs: processed / failed records + run config
# ---------------------------------------------------------------------------


class RunLogger:
    """Append-only CSV logs for processed and failed records plus a JSON
    snapshot of the run config."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.processed_path = out_dir / "processed_records.csv"
        self.failed_path = out_dir / "failed_records.csv"
        self._ensure_header(
            self.processed_path, ["subject_id", "study_id", "split"]
        )
        self._ensure_header(
            self.failed_path, ["subject_id", "study_id", "split", "error"]
        )

    @staticmethod
    def _ensure_header(path: Path, columns: list[str]) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(columns)

    def log_success(self, subject_id: str, study_id: str, split: str) -> None:
        with self.processed_path.open("a", newline="") as f:
            csv.writer(f).writerow([subject_id, study_id, split])

    def log_failure(
        self, subject_id: str, study_id: str, split: str, error: str
    ) -> None:
        with self.failed_path.open("a", newline="") as f:
            csv.writer(f).writerow([subject_id, study_id, split, error])

    def load_processed_keys(self) -> set[tuple[str, str]]:
        if not self.processed_path.exists():
            return set()
        keys: set[tuple[str, str]] = set()
        with self.processed_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((row["subject_id"], row["study_id"]))
        return keys

    def write_run_config(self, cfg: PipelineConfig) -> None:
        with (self.out_dir / "run_config.json").open("w") as f:
            json.dump(cfg.to_serializable(), f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _iter_worker_args(
    manifest: Iterable[ManifestEntry], cfg: PipelineConfig
) -> Iterator[tuple[str, str, str, PipelineConfig]]:
    for m in manifest:
        yield (m.subject_id, m.study_id, m.split, cfg)


def run_pipeline(cfg: PipelineConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _validate_runtime_dependencies()
    logger = RunLogger(cfg.out_dir)
    logger.write_run_config(cfg)

    LOG.info("Loading ids CSV: %s", cfg.ids_csv)
    pairs = load_ids_csv(cfg.ids_csv)
    LOG.info("Loaded %d unique (subject,study) pairs", len(pairs))

    manifest = build_manifest(
        pairs,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
        max_records=cfg.max_records,
    )
    LOG.info(
        "Manifest built: total=%d train=%d val=%d",
        len(manifest),
        sum(m.split == "train" for m in manifest),
        sum(m.split == "val" for m in manifest),
    )

    if cfg.resume:
        already = logger.load_processed_keys()
        before = len(manifest)
        manifest = [m for m in manifest if (m.subject_id, m.study_id) not in already]
        LOG.info(
            "Resume enabled: skipping %d already-processed records, %d remain",
            before - len(manifest),
            len(manifest),
        )

    writers = {
        "train": ChunkWriter("train", cfg.out_dir, cfg.chunk_size),
        "val": ChunkWriter("val", cfg.out_dir, cfg.chunk_size),
    }

    start = time.time()
    n_ok = 0
    n_fail = 0
    total = len(manifest)
    LOG.info("Launching worker pool with %d workers", cfg.num_workers)

    if cfg.num_workers <= 1:
        iterator = (_process_single(a) for a in _iter_worker_args(manifest, cfg))
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=cfg.num_workers)
        iterator = pool.imap_unordered(
            _process_single,
            _iter_worker_args(manifest, cfg),
            chunksize=8,
        )

    try:
        for i, res in enumerate(iterator, start=1):
            subject_id, study_id, split, status, arr, err = res
            if status == "ok" and arr is not None:
                writers[split].append(arr)
                logger.log_success(subject_id, study_id, split)
                n_ok += 1
            else:
                logger.log_failure(subject_id, study_id, split, err or "unknown")
                n_fail += 1

            if i % max(1, cfg.chunk_size) == 0 or i == total:
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                LOG.info(
                    "progress %d/%d ok=%d fail=%d rate=%.2f rec/s",
                    i,
                    total,
                    n_ok,
                    n_fail,
                    rate,
                )
    finally:
        if cfg.num_workers > 1:
            pool.close()
            pool.join()

    LOG.info("Finalizing split arrays")
    final_paths = {name: w.finalize() for name, w in writers.items()}
    for w in writers.values():
        w.cleanup_tmp()

    summary = {
        "total": total,
        "ok": n_ok,
        "failed": n_fail,
        "train_path": str(final_paths["train"]),
        "val_path": str(final_paths["val"]),
        "train_count": writers["train"].count,
        "val_count": writers["val"].count,
        "elapsed_s": time.time() - start,
    }
    with (cfg.out_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    LOG.info("Done: %s", summary)
    return summary
