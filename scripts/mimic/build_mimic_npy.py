"""CLI entrypoint for building the MIMIC-IV-ECG pretraining NPY files.

Usage (from repo root):

    python -m scripts.mimic.build_mimic_npy \
        --ids_csv ./mimic_ids.csv \
        --out_dir ./data/mimic \
        --num_workers 20 \
        --max_records 500

The script produces, under `--out_dir`:
    X_ecg_train.npy   # (N_train, 12, 1000) float32
    X_ecg_val.npy     # (N_val,   12, 1000) float32
    processed_records.csv
    failed_records.csv
    run_config.json
    run_summary.json

Pretraining only: no labels and no mean/std files are written; signals are raw
fp32 resampled from 500 Hz to 100 Hz.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

from .mimic_pipeline import (
    DEFAULT_BASE_URL,
    DEFAULT_PN_DIR_ROOT,
    PipelineConfig,
    run_pipeline,
)


def _default_workers() -> int:
    # Leave one core free for the parent process.
    n = os.cpu_count() or 2
    return max(1, n - 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download MIMIC-IV-ECG records listed in an ids CSV, resample to "
            "100 Hz, and write (N,12,1000) float32 NPY arrays for JEPA "
            "pretraining (train/val only, no labels, no normalization)."
        )
    )
    parser.add_argument(
        "--ids_csv",
        type=Path,
        default=Path("/Users/shadyali/TS-JEPA/mimic_ids.csv"),
        help="CSV with columns subject_id,study_id",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/Users/shadyali/TS-JEPA/data/mimic"),
        help="Output directory for X_ecg_{train,val}.npy and run logs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=_default_workers(),
        help="Number of parallel download+decode workers (default: CPU-1)",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Process at most this many records (default: all).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Fraction of subjects assigned to the validation split.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tmp_dir",
        type=Path,
        default=None,
        help="Optional override for local cache directory. Currently unused "
        "because we stream directly via wfdb; reserved for future backends.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="Base URL for the PhysioNet files tree.",
    )
    parser.add_argument(
        "--pn_dir_root",
        type=str,
        default=DEFAULT_PN_DIR_ROOT,
        help="Prefix passed to wfdb.rdrecord(pn_dir=...) for PhysioNet streaming.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2000,
        help=(
            "How many records to buffer in memory per split before flushing to "
            "a temporary chunk .npy file on disk."
        ),
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_backoff_s", type=float, default=2.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in processed_records.csv.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _to_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        ids_csv=args.ids_csv,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        max_records=args.max_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
        base_url=args.base_url,
        pn_dir_root=args.pn_dir_root,
        chunk_size=args.chunk_size,
        resume=args.resume,
        retries=args.retries,
        retry_backoff_s=args.retry_backoff_s,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    cfg = _to_config(args)
    summary = run_pipeline(cfg)
    print(summary)
    return 0


if __name__ == "__main__":
    # `spawn` start method is safer cross-platform (macOS default) and is
    # explicitly requested inside the pipeline too; setting it here avoids
    # fork warnings on Linux when the parent has already imported numpy/wfdb.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    sys.exit(main())
