"""Shared utilities for the ECG-JEPA pretraining data pipeline.

Each per-dataset module (:mod:`scripts.pretrain_data.chapman_ningbo`,
:mod:`scripts.pretrain_data.code15`) yields ``(record_id, signal_8x2500)``
pairs to the orchestrator in :mod:`scripts.pretrain_data.build_pretrain_npy`.

This module owns the parts that are identical across datasets:

  * 8-lead extraction from a 12-lead source (drop the 4 derivable leads
    III, aVR, aVL, aVF; keep I, II, V1, V2, V3, V4, V5, V6).
  * Resampling from the source rate to 250 Hz over 10 s -> 2500 samples.
  * "Drop missing values" filter (NaN / Inf / all-zero leads).
  * Chunked, crash-safe NPY writer that buffers to ``_tmp/<split>/`` and
    finalizes into ``X_pretrain_{train,val}.npy`` on close.

All paper references below point to Kim 2026 (arxiv 2410.08559).
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

LOG = logging.getLogger("pretrain_data")

# Output geometry: paper uses 10 s @ 250 Hz, 8 leads.
TARGET_LEADS: int = 8
TARGET_RATE_HZ: int = 250
TARGET_DURATION_S: float = 10.0
TARGET_LENGTH: int = int(TARGET_RATE_HZ * TARGET_DURATION_S)  # 2500

# Standard 12-lead order in WFDB / CODE-15 / PhysioNet ECG-Arrhythmia files.
# We keep limb leads {I, II} and chest leads {V1..V6}; drop {III, aVR, aVL, aVF},
# which can be linearly recovered from I and II via Einthoven's law.
TWELVE_LEAD_ORDER: tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)
EIGHT_LEAD_KEEP: tuple[str, ...] = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
EIGHT_LEAD_INDICES: tuple[int, ...] = tuple(
    TWELVE_LEAD_ORDER.index(name) for name in EIGHT_LEAD_KEEP
)

_CHUNK_NPY_RE = re.compile(r"^chunk_(\d{5})\.npy$")


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def select_eight_leads(
    signal: np.ndarray,
    lead_names: Iterable[str] | None = None,
) -> np.ndarray:
    """Project a 12-lead signal down to the paper's 8 leads.

    Args:
        signal: ``(12, T)`` float array. If `lead_names` is supplied it is
            used to look up indices; otherwise the standard 12-lead order
            (:data:`TWELVE_LEAD_ORDER`) is assumed.
        lead_names: Optional iterable of 12 lead names matching the rows of
            ``signal``.

    Returns:
        ``(8, T)`` float32 array in the order :data:`EIGHT_LEAD_KEEP`.
    """
    if signal.ndim != 2 or signal.shape[0] < TARGET_LEADS:
        raise ValueError(
            f"select_eight_leads expects (>=8, T); got {signal.shape}"
        )
    if lead_names is None:
        if signal.shape[0] != 12:
            raise ValueError(
                f"select_eight_leads needs lead_names when input is not 12-lead; "
                f"got shape {signal.shape}"
            )
        idx = EIGHT_LEAD_INDICES
    else:
        names_upper = [str(n).strip().upper() for n in lead_names]
        try:
            idx = tuple(names_upper.index(name) for name in EIGHT_LEAD_KEEP)
        except ValueError as e:
            raise ValueError(
                f"Could not find one of {EIGHT_LEAD_KEEP} in lead_names {names_upper}"
            ) from e
    return np.ascontiguousarray(signal[list(idx), :], dtype=np.float32)


def resample_to_target(
    signal: np.ndarray, source_length: int
) -> np.ndarray:
    """Resample along the time axis (last axis) to :data:`TARGET_LENGTH`.

    Uses ``scipy.signal.resample`` (Fourier method). This is the same routine
    the existing MIMIC pipeline uses; it produces clean results when the
    source signal is already band-limited (true for clinical ECG hardware).
    """
    from scipy.signal import resample

    if signal.shape[-1] != source_length:
        raise ValueError(
            f"resample_to_target expected last dim {source_length}, got "
            f"{signal.shape[-1]}"
        )
    if source_length == TARGET_LENGTH:
        return np.asarray(signal, dtype=np.float32)
    out = resample(signal, TARGET_LENGTH, axis=-1)
    return np.asarray(out, dtype=np.float32)


def is_valid_record(signal: np.ndarray) -> tuple[bool, str]:
    """Apply the paper's "drop records with missing values" filter.

    A record is rejected if any element is non-finite (NaN / +-Inf), or if any
    lead is entirely zero (which is how WFDB-encoded MIMIC/CODE-15 records
    represent disconnected leads).

    Returns:
        ``(ok, reason)``. ``reason`` is empty when ``ok`` is True.
    """
    if not np.isfinite(signal).all():
        return False, "non_finite"
    # Per-lead all-zero check: a real 10 s lead is essentially never identically zero.
    lead_max = np.abs(signal).max(axis=-1)
    if np.any(lead_max == 0):
        return False, "lead_all_zero"
    return True, ""


# ---------------------------------------------------------------------------
# Chunked NPY writer (split-aware, crash-safe, resume-friendly)
# ---------------------------------------------------------------------------


class ChunkWriter:
    """Buffers per-record arrays for one split and flushes to chunk files.

    Final filename on ``close()`` is ``X_pretrain_<split>.npy`` under
    ``out_dir``. The chunk directory ``<out_dir>/_tmp/<split>/`` is retained
    after finalization so the user can re-run the merge step without
    redownloading anything.
    """

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
        indices: list[int] = []
        for p in self.tmp_dir.glob("chunk_*.npy"):
            m = _CHUNK_NPY_RE.match(p.name)
            if m:
                indices.append(int(m.group(1)))
        return max(indices) + 1 if indices else 0

    def _count_from_existing(self) -> int:
        total = 0
        for p in sorted(self.tmp_dir.glob("chunk_*.npy")):
            if _CHUNK_NPY_RE.match(p.name):
                total += int(np.load(p, mmap_mode="r").shape[0])
        return total

    @property
    def count(self) -> int:
        return self._count + len(self._buf)

    def append(self, arr: np.ndarray) -> None:
        if arr.shape != (TARGET_LEADS, TARGET_LENGTH):
            raise ValueError(
                f"ChunkWriter expected ({TARGET_LEADS},{TARGET_LENGTH}); got {arr.shape}"
            )
        self._buf.append(arr.astype(np.float32, copy=False))
        if len(self._buf) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        chunk = np.stack(self._buf, axis=0).astype(np.float32, copy=False)
        path = self.tmp_dir / f"chunk_{self._next_chunk_idx:05d}.npy"
        tmp_path = path.with_name(f"{path.stem}.partial{path.suffix}")
        np.save(tmp_path, chunk)
        os.replace(tmp_path, path)
        self._count += chunk.shape[0]
        self._next_chunk_idx += 1
        self._buf.clear()

    def finalize(self) -> Path:
        self.flush()
        chunk_paths = sorted(
            p for p in self.tmp_dir.glob("chunk_*.npy") if _CHUNK_NPY_RE.match(p.name)
        )
        final_path = self.out_dir / f"X_pretrain_{self.split}.npy"

        if not chunk_paths:
            empty = np.empty((0, TARGET_LEADS, TARGET_LENGTH), dtype=np.float32)
            np.save(final_path, empty)
            return final_path

        total = sum(int(np.load(p, mmap_mode="r").shape[0]) for p in chunk_paths)
        tmp_final = final_path.with_suffix(".npy.partial")
        out = np.lib.format.open_memmap(
            tmp_final, mode="w+", dtype=np.float32,
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
        return final_path

    def cleanup_tmp(self) -> None:
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------


class RunLogger:
    """Append-only CSVs of processed/failed records, plus a JSON config dump."""

    def __init__(self, out_dir: Path, run_name: str):
        self.out_dir = out_dir
        self.processed_path = out_dir / f"processed_{run_name}.csv"
        self.failed_path = out_dir / f"failed_{run_name}.csv"
        self._ensure_header(self.processed_path, ["dataset", "record_id", "split"])
        self._ensure_header(
            self.failed_path, ["dataset", "record_id", "split", "reason"]
        )

    @staticmethod
    def _ensure_header(path: Path, columns: list[str]) -> None:
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(columns)

    def log_success(self, dataset: str, record_id: str, split: str) -> None:
        with self.processed_path.open("a", newline="") as f:
            csv.writer(f).writerow([dataset, record_id, split])

    def log_failure(
        self, dataset: str, record_id: str, split: str, reason: str
    ) -> None:
        with self.failed_path.open("a", newline="") as f:
            csv.writer(f).writerow([dataset, record_id, split, reason])

    def load_processed_keys(self) -> set[tuple[str, str]]:
        if not self.processed_path.exists():
            return set()
        keys: set[tuple[str, str]] = set()
        with self.processed_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys.add((row["dataset"], row["record_id"]))
        return keys

    def write_run_config(self, payload: dict) -> None:
        with (self.out_dir / "run_config.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Subject-level deterministic split
# ---------------------------------------------------------------------------


@dataclass
class SplitConfig:
    val_ratio: float = 0.05
    seed: int = 0


def assign_split(record_id: str, split_cfg: SplitConfig) -> str:
    """Stable per-record split assignment.

    A 64-bit hash of ``record_id`` (``hash`` is randomized per-process by
    PYTHONHASHSEED, so we use ``int.from_bytes(md5(...))`` for determinism)
    is taken modulo 1e6 and compared against the val cutoff. This avoids
    needing a global manifest of every record up front.
    """
    import hashlib

    h = hashlib.md5(f"{split_cfg.seed}|{record_id}".encode()).digest()
    bucket = int.from_bytes(h[:4], byteorder="big") % 1_000_000
    cutoff = int(round(split_cfg.val_ratio * 1_000_000))
    return "val" if bucket < cutoff else "train"


def to_serializable(obj) -> dict:
    if hasattr(obj, "to_serializable"):
        return obj.to_serializable()
    if hasattr(obj, "__dataclass_fields__"):
        d = asdict(obj)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
        return d
    return dict(obj)
