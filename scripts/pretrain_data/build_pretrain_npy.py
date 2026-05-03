"""CLI orchestrator for the ECG-JEPA pretraining data pipeline.

Combines Chapman + Ningbo (PhysioNet ``ecg-arrhythmia/1.0.0``) and CODE-15%
(Zenodo 4916206) into the unified pair of arrays consumed by
:class:`src.data.pretrain_dataset.PretrainECGDataset`:

    <out_dir>/X_pretrain_train.npy   # (N_train, 8, 2500) float32
    <out_dir>/X_pretrain_val.npy     # (N_val,   8, 2500) float32

All records are 8-lead (I, II, V1-V6), resampled to 250 Hz, 10 s long.
Records with NaN, Inf, or any all-zero lead are dropped (matching the
paper's "exclude recordings with missing values" criterion).

Splits are decided per record using a deterministic hash of ``record_id``
into 1e6 buckets vs ``--val_ratio`` (default 5%).

Usage::

    # Chapman + Ningbo only (smaller; ~43k records after filtering):
    python -m scripts.pretrain_data.build_pretrain_npy \\
        --datasets chapman_ningbo \\
        --out_dir data/pretrain --val_ratio 0.05

    # Both datasets (paper-faithful; ~170k records):
    python -m scripts.pretrain_data.build_pretrain_npy \\
        --datasets chapman_ningbo code15 \\
        --out_dir data/pretrain \\
        --code15_cache /scratch/$USER/code15 \\
        --val_ratio 0.05

    # Smoke test (a few CODE-15 records from one shard):
    python -m scripts.pretrain_data.build_pretrain_npy \\
        --datasets code15 \\
        --code15_shards 0 --code15_max_per_shard 50 \\
        --out_dir data/pretrain_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .chapman_ningbo import (
    DATASET_NAME as CHAPMAN_NAME,
    ChapmanNingboConfig,
    iter_processed_records as iter_chapman_records,
)
from .code15 import (
    DATASET_NAME as CODE15_NAME,
    Code15Config,
    SHARD_INDICES as CODE15_SHARDS,
    iter_processed_records as iter_code15_records,
)
from .common import (
    ChunkWriter,
    RunLogger,
    SplitConfig,
    assign_split,
    to_serializable,
)


KNOWN_DATASETS: tuple[str, ...] = (CHAPMAN_NAME, CODE15_NAME)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build paper-faithful X_pretrain_{train,val}.npy from "
                    "Chapman+Ningbo and/or CODE-15."
    )
    p.add_argument(
        "--datasets", nargs="+", default=list(KNOWN_DATASETS),
        choices=list(KNOWN_DATASETS),
        help="Which sources to include (default: both).",
    )
    p.add_argument("--out_dir", type=Path, default=Path("data/pretrain"))
    p.add_argument(
        "--val_ratio", type=float, default=0.05,
        help="Fraction of records assigned to the validation split.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--chunk_size", type=int, default=2000,
        help="Records buffered in memory per split before flushing to disk.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip records already listed in processed_<dataset>.csv.",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    # Chapman + Ningbo knobs.
    p.add_argument(
        "--chapman_max_records", type=int, default=None,
        help="Smoke test: limit Chapman+Ningbo to N records.",
    )
    p.add_argument("--chapman_retries", type=int, default=3)
    p.add_argument("--chapman_retry_backoff_s", type=float, default=2.0)

    # CODE-15 knobs.
    p.add_argument(
        "--code15_cache", type=Path, default=Path("data/_code15_cache"),
        help="Where to download zips/HDF5s. On MSI use /scratch.",
    )
    p.add_argument(
        "--code15_shards", type=int, nargs="+", default=list(CODE15_SHARDS),
        help="Shard indices (0-17) to process; default = all 18.",
    )
    p.add_argument(
        "--code15_keep_downloads", action="store_true",
        help="Keep the zip+HDF5 around after processing each shard.",
    )
    p.add_argument(
        "--code15_max_per_shard", type=int, default=None,
        help="Smoke test: limit each CODE-15 shard to N records.",
    )
    p.add_argument("--code15_retries", type=int, default=3)
    p.add_argument("--code15_retry_backoff_s", type=float, default=5.0)
    return p


def _make_chapman_cfg(args) -> ChapmanNingboConfig:
    return ChapmanNingboConfig(
        max_records=args.chapman_max_records,
        retries=args.chapman_retries,
        retry_backoff_s=args.chapman_retry_backoff_s,
    )


def _make_code15_cfg(args) -> Code15Config:
    return Code15Config(
        cache_dir=args.code15_cache,
        shards=tuple(args.code15_shards),
        keep_downloads=args.code15_keep_downloads,
        retries=args.code15_retries,
        retry_backoff_s=args.code15_retry_backoff_s,
        max_records_per_shard=args.code15_max_per_shard,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    LOG = logging.getLogger("pretrain_data.build")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = SplitConfig(val_ratio=args.val_ratio, seed=args.seed)
    writers = {
        "train": ChunkWriter("train", args.out_dir, args.chunk_size),
        "val": ChunkWriter("val", args.out_dir, args.chunk_size),
    }

    runlog = RunLogger(args.out_dir, run_name="pretrain")
    runlog.write_run_config({
        "datasets": list(args.datasets),
        "out_dir": str(args.out_dir),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "resume": args.resume,
        "chapman_ningbo": to_serializable(_make_chapman_cfg(args)),
        "code15": to_serializable(_make_code15_cfg(args)),
    })

    skip_ids: set[str] = set()
    if args.resume:
        for (_dataset, rid) in runlog.load_processed_keys():
            skip_ids.add(rid)
        LOG.info("Resume mode: skipping %d already-processed records", len(skip_ids))

    counts = {ds: {"ok": 0, "fail": 0} for ds in args.datasets}
    start = time.time()

    for ds in args.datasets:
        LOG.info("=== Processing dataset: %s ===", ds)
        if ds == CHAPMAN_NAME:
            iterator = iter_chapman_records(_make_chapman_cfg(args), skip_ids=skip_ids)
        elif ds == CODE15_NAME:
            iterator = iter_code15_records(_make_code15_cfg(args), skip_ids=skip_ids)
        else:
            raise ValueError(f"unknown dataset {ds!r}")

        for i, (record_id, sig, status) in enumerate(iterator, start=1):
            if status != "ok" or sig.size == 0:
                runlog.log_failure(ds, record_id, "n/a", status)
                counts[ds]["fail"] += 1
            else:
                split = assign_split(record_id, split_cfg)
                writers[split].append(sig)
                runlog.log_success(ds, record_id, split)
                counts[ds]["ok"] += 1

            if i % 500 == 0:
                elapsed = time.time() - start
                LOG.info(
                    "[%s] processed=%d ok=%d fail=%d (%.2f rec/s)",
                    ds, i, counts[ds]["ok"], counts[ds]["fail"],
                    i / max(elapsed, 1e-9),
                )

    LOG.info("Finalizing split arrays...")
    final_paths = {name: w.finalize() for name, w in writers.items()}
    for w in writers.values():
        if not args.resume:
            # In resume mode the user may want to re-run finalization later.
            w.cleanup_tmp()

    summary = {
        "datasets": list(args.datasets),
        "counts": counts,
        "train_path": str(final_paths["train"]),
        "val_path": str(final_paths["val"]),
        "train_count": writers["train"].count,
        "val_count": writers["val"].count,
        "elapsed_s": time.time() - start,
    }
    with (args.out_dir / "run_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    LOG.info("Done: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
