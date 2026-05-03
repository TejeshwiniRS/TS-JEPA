"""CODE-15% (Zenodo 4916206) loader.

CODE-15% is the second pretraining source in the ECG-JEPA paper (143,328
of its 345,779 records are 10 s long; after dropping records with missing
values the paper keeps ~130,900). The dataset is distributed as 18 zipped
HDF5 shards on Zenodo, each ~2.7 GB compressed:

    https://zenodo.org/records/4916206/files/exams_part{0..17}.zip

Each ``exams_part{i}.hdf5`` contains:
  - ``tracings``  : float, shape ``(N, 4096, 12)``, 400 Hz, mV.
  - ``exam_id``   : int,   shape ``(N,)``.

Lead order: ``I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6``.

Signals are padded with zeros on both sides to 4096 samples (10 s = 4000
samples, 7 s = 2800 samples). The paper uses only the 10 s subset; we
detect length by counting leading/trailing all-zero rows and require the
core signal to be at least 4000 samples long.

The loader processes one shard at a time and (by default) deletes the
download/extracted files once they're consumed, so disk usage stays bounded
to one shard at a time even for the full dataset.
"""

from __future__ import annotations

import logging
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .common import (
    TARGET_LEADS,
    TARGET_LENGTH,
    is_valid_record,
    resample_to_target,
    select_eight_leads,
)

LOG = logging.getLogger("pretrain_data.code15")

DATASET_NAME: str = "code15"
ZENODO_RECORD: str = "4916206"
DEFAULT_BASE_URL: str = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
SHARD_INDICES: tuple[int, ...] = tuple(range(18))

SOURCE_RATE_HZ: int = 400
HDF5_PADDED_LENGTH: int = 4096
TEN_SECOND_SAMPLES: int = SOURCE_RATE_HZ * 10  # 4000


@dataclass
class Code15Config:
    base_url: str = DEFAULT_BASE_URL
    cache_dir: Path = Path("data/_code15_cache")
    shards: tuple[int, ...] = SHARD_INDICES
    keep_downloads: bool = False     # delete zip+hdf5 after processing each shard
    retries: int = 3
    retry_backoff_s: float = 5.0
    read_chunk: int = 1024           # records per HDF5 read
    max_records_per_shard: Optional[int] = None  # for smoke tests


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dst: Path, retries: int, backoff: float) -> Path:
    """Download a file with retries. Skips if `dst` already exists with size>0."""
    if dst.exists() and dst.stat().st_size > 0:
        LOG.info("Reusing existing download %s (%.1f GB)",
                 dst, dst.stat().st_size / 1e9)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".partial")
    last_err: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            LOG.info("Downloading %s -> %s (attempt %d/%d)",
                     url, dst, attempt, retries)
            with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MB chunks
            tmp.replace(dst)
            LOG.info("Downloaded %s (%.1f GB)", dst, dst.stat().st_size / 1e9)
            return dst
        except BaseException as e:  # noqa: BLE001
            last_err = e
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(backoff * attempt)
            else:
                raise
    raise RuntimeError(f"download failed after {retries} retries: {last_err}")


def _unzip_one(zip_path: Path, out_dir: Path) -> Path:
    """Extract a single-file zip and return the path to the extracted file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if not names:
            raise RuntimeError(f"{zip_path} is empty")
        if len(names) != 1:
            LOG.warning("%s contains %d files; using first", zip_path, len(names))
        member = names[0]
        # Strip any internal directory parts -> flat output filename.
        out_name = Path(member).name
        target = out_dir / out_name
        if target.exists() and target.stat().st_size > 0:
            return target
        tmp = target.with_suffix(target.suffix + ".partial")
        with zf.open(member) as src, tmp.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
        tmp.replace(target)
        return target


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------


def _detect_signal_length(tracing_4096x12: np.ndarray) -> tuple[int, int]:
    """Find where the real signal starts/ends inside the zero-padded record.

    Returns ``(start, stop)`` such that ``tracing[start:stop]`` is the
    non-padded portion. ``stop - start`` will be ~4000 for a 10 s record
    and ~2800 for a 7 s record.
    """
    n = tracing_4096x12.shape[0]
    r = 0
    while r < n and np.all(tracing_4096x12[r, :] == 0):
        r += 1
    s = n
    while s > r and np.all(tracing_4096x12[s - 1, :] == 0):
        s -= 1
    return r, s


def _process_one(
    tracing_4096x12: np.ndarray,
    min_core_samples: int,
) -> tuple[Optional[np.ndarray], str]:
    """Project to 8 leads, drop 7 s records, resample to 250 Hz/2500 samples."""
    start, stop = _detect_signal_length(tracing_4096x12)
    core_len = stop - start
    if core_len < min_core_samples:
        return None, f"too_short:{core_len}"

    # Trim padding then take the leading 4000 samples (== 10 s @ 400 Hz).
    core = tracing_4096x12[start:stop, :]
    if core.shape[0] >= TEN_SECOND_SAMPLES:
        core = core[:TEN_SECOND_SAMPLES, :]
    else:
        # Right-pad with the trailing edge value? No - paper drops these.
        return None, f"non_ten_sec:{core.shape[0]}"

    # Transpose to (12, T) and select the 8 paper-faithful leads.
    sig12 = np.ascontiguousarray(core.T, dtype=np.float32)  # (12, 4000)
    try:
        sig8 = select_eight_leads(sig12)
    except ValueError as e:
        return None, f"lead_select:{e}"

    ok, reason = is_valid_record(sig8)
    if not ok:
        return None, reason

    sig8 = resample_to_target(sig8, source_length=TEN_SECOND_SAMPLES)
    if sig8.shape != (TARGET_LEADS, TARGET_LENGTH):
        return None, f"final_shape:{sig8.shape}"
    return sig8, "ok"


# ---------------------------------------------------------------------------
# Public iterator
# ---------------------------------------------------------------------------


def iter_processed_records(
    cfg: Code15Config,
    skip_ids: Optional[set[str]] = None,
    min_core_samples: int = 4000,
) -> Iterator[tuple[str, np.ndarray, str]]:
    """Yield ``(record_id, signal_8x2500, status_or_reason)`` over all shards.

    ``record_id`` is ``"code15_<exam_id>"``. ``status_or_reason`` is ``"ok"``
    for retained records, otherwise a short failure code.

    Each shard is downloaded, unzipped, processed, and (unless
    ``cfg.keep_downloads``) cleaned up before moving to the next shard.
    """
    import h5py

    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for shard_idx in cfg.shards:
        zip_url = f"{cfg.base_url}/exams_part{shard_idx}.zip"
        zip_path = cache_dir / f"exams_part{shard_idx}.zip"

        try:
            _download(zip_url, zip_path, cfg.retries, cfg.retry_backoff_s)
        except BaseException as e:  # noqa: BLE001
            LOG.error("Skipping shard %d: download error %r", shard_idx, e)
            continue

        try:
            hdf5_path = _unzip_one(zip_path, cache_dir)
        except BaseException as e:  # noqa: BLE001
            LOG.error("Skipping shard %d: unzip error %r", shard_idx, e)
            if not cfg.keep_downloads:
                zip_path.unlink(missing_ok=True)
            continue

        LOG.info("Processing shard %d -> %s", shard_idx, hdf5_path)
        try:
            with h5py.File(hdf5_path, "r") as f:
                tracings = f["tracings"]    # (N, 4096, 12)
                exam_ids = f["exam_id"]     # (N,)
                n_total = tracings.shape[0]

                if cfg.max_records_per_shard is not None:
                    n_total = min(n_total, cfg.max_records_per_shard)

                LOG.info("Shard %d: %d records to process", shard_idx, n_total)

                for start in range(0, n_total, cfg.read_chunk):
                    stop = min(start + cfg.read_chunk, n_total)
                    chunk = np.asarray(tracings[start:stop])
                    ids_chunk = np.asarray(exam_ids[start:stop])

                    for j in range(chunk.shape[0]):
                        record_id = f"code15_{int(ids_chunk[j])}"
                        if skip_ids and record_id in skip_ids:
                            continue
                        sig, status = _process_one(
                            chunk[j], min_core_samples=min_core_samples,
                        )
                        if sig is None:
                            yield record_id, np.empty((0,), dtype=np.float32), status
                        else:
                            yield record_id, sig, "ok"
        finally:
            if not cfg.keep_downloads:
                zip_path.unlink(missing_ok=True)
                hdf5_path.unlink(missing_ok=True)
                LOG.info(
                    "Cleaned shard %d cache (set --code15_keep_downloads to retain)",
                    shard_idx,
                )
