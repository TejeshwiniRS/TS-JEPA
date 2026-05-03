"""Chapman + Ningbo (PhysioNet ``ecg-arrhythmia/1.0.0``) loader.

The "ecg-arrhythmia" PhysioNet database is the unified Chapman-Shaoxing +
Ningbo collection used in the ECG-JEPA paper as its first pretraining source
(45,152 12-lead 10 s ECGs at 500 Hz). The records live under

    physionet.org/files/ecg-arrhythmia/1.0.0/WFDBRecords/<group>/<subgroup>/<JSxxxxx>.{hea,mat}

with ``RECORDS`` index files at **two** levels: the root ``RECORDS`` lists
452 subdirectories under ``WFDBRecords/``; each subdirectory has its own
``RECORDS`` listing the ``JSxxxxx`` WFDB base names. See
:func:`list_record_paths`.
with ``pn_dir=`` to stream each record without saving the raw WFDB on disk;
this matches the existing MIMIC pipeline pattern used elsewhere in the repo.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .common import (
    TARGET_LEADS,
    TARGET_LENGTH,
    is_valid_record,
    resample_to_target,
    select_eight_leads,
)

LOG = logging.getLogger("pretrain_data.chapman_ningbo")

DATASET_NAME: str = "chapman_ningbo"
PN_DIR_ROOT: str = "ecg-arrhythmia/1.0.0"
SOURCE_RATE_HZ: int = 500
SOURCE_LENGTH: int = SOURCE_RATE_HZ * 10  # 5000


@dataclass
class ChapmanNingboConfig:
    pn_dir_root: str = PN_DIR_ROOT
    retries: int = 3
    retry_backoff_s: float = 2.0
    max_records: Optional[int] = None  # None = all


def list_record_paths(pn_dir_root: str = PN_DIR_ROOT) -> list[str]:
    """Return every WFDB base-name path relative to ``pn_dir_root``.

    The top-level ``RECORDS`` for ``ecg-arrhythmia/1.0.0`` does **not** list
    individual exams. It lists 452 **subdirectories** (one per 100-patient
    bucket), each ending with a trailing slash::

        WFDBRecords/01/010/
        WFDBRecords/01/011/
        ...

    Each subdirectory has its own small ``RECORDS`` file that lists base
    names only (``JS00001``, ``JS00002``, ...). We concatenate those into
    full relative paths::

        WFDBRecords/01/010/JS00001
        WFDBRecords/01/010/JS00002
        ...

    which is what :func:`_fetch_record` expects.
    """
    import wfdb

    LOG.info("Fetching top-level RECORDS from PhysioNet (%s)...", pn_dir_root)
    top_entries = wfdb.io.get_record_list(pn_dir_root)
    # Normalize "WFDBRecords/01/010/" -> "WFDBRecords/01/010"
    normalized = [e.strip().rstrip("/") for e in top_entries if e.strip()]

    all_paths: list[str] = []
    n_top = len(normalized)
    for i, entry in enumerate(normalized):
        if i % 50 == 0 or i == n_top - 1:
            LOG.info("Expanding subdirectory %d / %d (%s)...", i + 1, n_top, entry)
        sub_db = f"{pn_dir_root}/{entry}"
        try:
            basenames = wfdb.io.get_record_list(sub_db)
        except Exception as e:  # noqa: BLE001 - one bad folder should not kill the run
            LOG.warning("Skipping %s (could not read RECORDS: %s)", sub_db, e)
            continue
        for base in basenames:
            b = base.strip()
            if not b or b.startswith("#"):
                continue
            all_paths.append(f"{entry}/{b}")

    LOG.info("Expanded to %d individual WFDB records under %s", len(all_paths), pn_dir_root)
    return all_paths


def _split_pn_path(record_path: str) -> tuple[str, str]:
    """``"WFDBRecords/01/010/JS00001"`` -> (``"WFDBRecords/01/010"``, ``"JS00001"``)."""
    if "/" not in record_path:
        return "", record_path
    head, tail = record_path.rsplit("/", 1)
    return head, tail


def _fetch_record(
    record_path: str,
    cfg: ChapmanNingboConfig,
) -> tuple[np.ndarray, list[str]]:
    """Stream one Chapman/Ningbo record. Returns (signal_12xT, lead_names)."""
    import wfdb

    head, name = _split_pn_path(record_path)
    pn_dir = f"{cfg.pn_dir_root}/{head}" if head else cfg.pn_dir_root

    last_err: BaseException | None = None
    for attempt in range(1, cfg.retries + 1):
        try:
            rec = wfdb.rdrecord(name, pn_dir=pn_dir)
            sig = rec.p_signal  # (T, C)
            if sig is None:
                raise RuntimeError("wfdb.rdrecord returned empty p_signal")
            sig = np.asarray(sig, dtype=np.float32).T  # -> (C, T)
            return sig, list(rec.sig_name)
        except BaseException as e:  # noqa: BLE001 - network-level retry
            last_err = e
            if attempt < cfg.retries:
                time.sleep(cfg.retry_backoff_s * attempt)
            else:
                raise
    raise RuntimeError(f"failed after {cfg.retries} retries: {last_err}")


def iter_processed_records(
    cfg: ChapmanNingboConfig,
    skip_ids: Optional[set[str]] = None,
) -> Iterator[tuple[str, np.ndarray, str]]:
    """Yield ``(record_id, signal_8x2500, status_or_reason)`` per record.

    ``status_or_reason`` is ``"ok"`` for healthy records, otherwise a short
    failure code (``"non_finite"``, ``"lead_all_zero"``, ``"shape:..."``,
    ``"download:..."``). The orchestrator decides how to log.

    If ``skip_ids`` is given, records whose id is already in the set are
    skipped entirely (used for resuming an interrupted run).
    """
    record_paths = list_record_paths(cfg.pn_dir_root)
    if cfg.max_records is not None:
        record_paths = record_paths[: cfg.max_records]

    for record_path in record_paths:
        # Full relative path is unique (basename JSxxxxx repeats across folders).
        record_id = record_path
        if skip_ids and record_id in skip_ids:
            continue
        try:
            sig, lead_names = _fetch_record(record_path, cfg)
        except BaseException as e:  # noqa: BLE001
            LOG.debug("download failed for %s: %r", record_id, e)
            yield record_id, np.empty((0,), dtype=np.float32), f"download:{type(e).__name__}"
            continue

        if sig.shape[0] < 12 or sig.shape[1] != SOURCE_LENGTH:
            yield record_id, np.empty((0,), dtype=np.float32), f"shape:{sig.shape}"
            continue
        try:
            sig8 = select_eight_leads(sig, lead_names=lead_names)
        except ValueError as e:
            yield record_id, np.empty((0,), dtype=np.float32), f"lead_select:{e}"
            continue

        ok, reason = is_valid_record(sig8)
        if not ok:
            yield record_id, np.empty((0,), dtype=np.float32), reason
            continue

        try:
            sig8 = resample_to_target(sig8, source_length=SOURCE_LENGTH)
        except Exception as e:  # noqa: BLE001
            yield record_id, np.empty((0,), dtype=np.float32), f"resample:{e}"
            continue

        if sig8.shape != (TARGET_LEADS, TARGET_LENGTH):
            yield record_id, np.empty((0,), dtype=np.float32), f"final_shape:{sig8.shape}"
            continue

        yield record_id, sig8, "ok"
