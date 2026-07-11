#!/usr/bin/env python3
"""Verify MMHS dataset layout for Stage-1 training.

Exit 0 if all *required* files pass.
Exit 1 if any required check fails.

Usage:
  python scripts/verify_dataset.py
  python scripts/verify_dataset.py --dest /path/to/dataset
  python scripts/verify_dataset.py --strict   # also require captions + image store
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [verify] {msg}", flush=True)


def ok(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [verify] ✓ {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [verify] ✗ {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [verify] ! {msg}", flush=True)


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def check_file(path: Path, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if not path.is_file():
        return False, "not a file"
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"too small ({human_size(size)})"
    return True, human_size(size)


def check_dir(path: Path, min_entries: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if not path.is_dir():
        return False, "not a directory"
    # count without loading all names into huge list if possible
    n = 0
    for _ in path.iterdir():
        n += 1
        if n >= min_entries and min_entries > 0:
            # still try a quick full count for reporting when small min
            break
    # full count for report (img_resized is large — sample estimate)
    if path.name == "img_resized":
        # faster: count only .jpg via scandir
        n = sum(1 for e in path.iterdir() if e.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if n < min_entries:
            return False, f"only {n} images (need ≥{min_entries})"
        return True, f"{n:,} images"
    n = sum(1 for _ in path.iterdir())
    if n < min_entries:
        return False, f"only {n} entries (need ≥{min_entries})"
    return True, f"{n} entries"


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify MMHS dataset for training")
    ap.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Dataset dir (default: MMHS_DATA_DIR or ./dataset)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Require captions + image store + full image count",
    )
    ap.add_argument(
        "--min-images",
        type=int,
        default=100_000,
        help="Minimum images under img_resized (default 100000)",
    )
    args = ap.parse_args()

    import os

    if args.dest is not None:
        dataset_dir = Path(args.dest)
    elif os.environ.get("MMHS_DATA_DIR"):
        dataset_dir = Path(os.environ["MMHS_DATA_DIR"])
    else:
        dataset_dir = PROJECT_ROOT / "dataset"

    results_dir = PROJECT_ROOT / "results"
    log(f"Project root : {PROJECT_ROOT}")
    log(f"Dataset dir  : {dataset_dir}")
    log(f"Strict mode  : {args.strict}")

    required_files = [
        ("MMHS150K_GT.json", 1_000_000),       # large JSON
        ("processed_labels.json", 100_000),
        ("ocr_filtered.json", 10_000),
    ]
    # OCR fallback is optional if filtered exists
    optional_files = [
        ("ocr_consolidated.json", 1_000),
        ("ocr_consolidated_filtered.json", 1_000),
        ("MMHS150K_readme.txt", 10),
        ("hatespeech_keywords.txt", 10),
        ("image_store_224.bin", 1_000_000),
        ("image_store_224_index.json", 1_000),
        ("vlm_captions.json", 1_000),
    ]
    required_dirs = [
        ("img_resized", args.min_images if args.strict else 1_000),
        ("splits", 3),
    ]
    required_split_files = [
        "train_ids.txt",
        "val_ids.txt",
        "test_ids.txt",
    ]

    failures = 0
    warnings = 0

    log("── Required files ──────────────────────────────────")
    for name, min_b in required_files:
        path = dataset_dir / name
        good, info = check_file(path, min_b)
        if good:
            ok(f"{name}  ({info})")
        else:
            fail(f"{name}  → {info}")
            failures += 1

    log("── Required directories ─────────────────────────────")
    for name, min_n in required_dirs:
        path = dataset_dir / name
        good, info = check_dir(path, min_n)
        if good:
            ok(f"{name}/  ({info})")
        else:
            fail(f"{name}/  → {info}")
            failures += 1

    log("── Split files ─────────────────────────────────────")
    for name in required_split_files:
        path = dataset_dir / "splits" / name
        good, info = check_file(path, 100)
        if good:
            # count lines
            try:
                n_lines = sum(1 for _ in open(path, "r", encoding="utf-8") if _.strip())
                ok(f"splits/{name}  ({info}, {n_lines:,} ids)")
            except Exception as e:
                fail(f"splits/{name}  unreadable: {e}")
                failures += 1
        else:
            fail(f"splits/{name}  → {info}")
            failures += 1

    log("── Optional / recommended ──────────────────────────")
    for name, min_b in optional_files:
        path = dataset_dir / name
        good, info = check_file(path, min_b)
        if good:
            ok(f"{name}  ({info})")
        else:
            warn(f"{name}  → {info}")
            warnings += 1

    # Captions: results/ or dataset/
    cap_paths = [
        results_dir / "vlm_captions.json",
        dataset_dir / "vlm_captions.json",
    ]
    cap_found = None
    for cp in cap_paths:
        good, info = check_file(cp, 1_000)
        if good:
            cap_found = cp
            try:
                rel = cp.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = cp
            ok(f"captions: {rel} ({info})")
            break
    if cap_found is None:
        msg = "vlm_captions.json missing (results/ or dataset/) — use text_mode=tweet_ocr or copy captions"
        if args.strict:
            fail(msg)
            failures += 1
        else:
            warn(msg)
            warnings += 1

    # Content sanity: GT + labels keys
    log("── Content sanity checks ───────────────────────────")
    gt_path = dataset_dir / "MMHS150K_GT.json"
    labels_path = dataset_dir / "processed_labels.json"
    if gt_path.exists() and labels_path.exists():
        try:
            log("Loading GT JSON (may take ~30–60s)…")
            t0 = time.time()
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
            log(f"GT loaded in {time.time()-t0:.1f}s  keys={len(gt):,}")

            log("Loading processed_labels.json…")
            t0 = time.time()
            with open(labels_path, "r", encoding="utf-8") as f:
                labels = json.load(f)
            log(f"Labels loaded in {time.time()-t0:.1f}s  keys={len(labels):,}")

            if len(gt) < 100_000:
                fail(f"GT has only {len(gt):,} entries (expected ~150k)")
                failures += 1
            else:
                ok(f"GT entry count {len(gt):,}")

            # Sample 3 IDs from train split
            train_ids_path = dataset_dir / "splits" / "train_ids.txt"
            if train_ids_path.exists():
                with open(train_ids_path, "r") as f:
                    sample_ids = [ln.strip() for ln in f if ln.strip()][:5]
                missing_gt = [i for i in sample_ids if i not in gt]
                missing_lab = [i for i in sample_ids if i not in labels]
                if missing_gt:
                    fail(f"Sample train IDs missing from GT: {missing_gt}")
                    failures += 1
                else:
                    ok(f"Sample train IDs present in GT ({len(sample_ids)} checked)")
                if missing_lab:
                    fail(f"Sample train IDs missing from labels: {missing_lab}")
                    failures += 1
                else:
                    ok(f"Sample train IDs present in labels")

                # Soft label schema
                lab0 = labels[sample_ids[0]]
                for key in ("hard_label_binary", "soft_label_binary", "agreement_level"):
                    if key not in lab0:
                        fail(f"labels missing field '{key}'")
                        failures += 1
                    else:
                        ok(f"labels field '{key}' present (e.g. {lab0[key]!r})")

                # Image exists for sample
                img_dir = dataset_dir / "img_resized"
                img_ok = 0
                for sid in sample_ids:
                    if (img_dir / f"{sid}.jpg").exists():
                        img_ok += 1
                if img_ok == 0:
                    fail("No sample train images found under img_resized/")
                    failures += 1
                else:
                    ok(f"{img_ok}/{len(sample_ids)} sample train images exist on disk")

        except Exception as e:
            fail(f"Content sanity failed: {e}")
            failures += 1
    else:
        warn("Skipping content sanity (GT or labels missing)")

    log("════════════════════════════════════════════════════")
    if failures == 0:
        ok(f"ALL REQUIRED CHECKS PASSED  (warnings={warnings})")
        if warnings:
            warn("Training can proceed; optional assets missing may reduce quality.")
        return 0
    fail(f"FAILED required checks: {failures}  (warnings={warnings})")
    fail("Fix dataset paths / re-run: python scripts/setup_kaggle_data.py --force")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
