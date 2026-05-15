"""Strict precheck for MMHS150K training runs.

Checks performed:
- required files and directories exist under the dataset root
- GT JSON is parseable and looks like a mapping
- split files exist and only reference GT IDs
- a few images can be opened
- OCR consolidated file is present or OCR directories exist

Exit codes:
 0 = ready for full training
 1 = warnings only
 2 = critical missing files or invalid split membership

Usage:
    python scripts/precheck_dataset.py
    python scripts/precheck_dataset.py --dataset-dir dataset
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Ensure repository root is on Python path so `src` package imports work
# This makes the script runnable as: `python scripts/precheck_dataset.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (
    DATASET_DIR,
    TOTAL_SAMPLES,
)


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load JSON {path}: {e}")
        return None


def check_file(p: Path, critical: bool = True):
    if not p.exists():
        msg = f"MISSING: {p}"
        if critical:
            print("[CRITICAL] " + msg)
            return False
        else:
            print("[WARN] " + msg)
            return True
    print(f"OK: {p}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_DIR))
    parser.add_argument(
        "--min-img-coverage",
        type=float,
        default=0.99,
        help="minimum fraction of split IDs that must have images",
    )
    parser.add_argument("--sample-check-images", type=int, default=5,
                        help="number of images to try opening for sanity")
    args = parser.parse_args()

    ds_root = Path(args.dataset_dir)
    gt_file = ds_root / "MMHS150K_GT.json"
    img_dir = ds_root / "img_resized"
    splits_dir = ds_root / "splits"
    ocr_consolidated = ds_root / "ocr_consolidated.json"
    ocr_dir_old = ds_root / "img_txt"
    ocr_dir_new = ds_root / "img_txt_new"
    processed_labels_file = ds_root / "processed_labels.json"

    print(f"Dataset root: {ds_root}")

    ok = True
    critical_fail = False

    # Files
    if not check_file(gt_file, critical=True):
        sys.exit(2)

    if not check_file(img_dir, critical=True):
        sys.exit(2)

    if not check_file(splits_dir, critical=True):
        sys.exit(2)

    # Splits files
    split_files = {
        "train": splits_dir / "train_ids.txt",
        "val": splits_dir / "val_ids.txt",
        "test": splits_dir / "test_ids.txt",
    }
    for name, p in split_files.items():
        if not p.exists():
            print(f"[CRITICAL] Missing split file: {p}")
            critical_fail = True
        else:
            print(f"OK: split {name} -> {p}")

    if critical_fail:
        print("Critical split files missing — aborting.")
        sys.exit(2)

    # Load GT
    gt = load_json(gt_file)
    if gt is None or not isinstance(gt, dict):
        print("[CRITICAL] GT JSON invalid or not a dict")
        sys.exit(2)

    n_gt = len(gt)
    print(f"GT entries: {n_gt} (expected approx {TOTAL_SAMPLES})")
    if n_gt != TOTAL_SAMPLES:
        print(
            f"[CRITICAL] GT size mismatch: found {n_gt}, expected {TOTAL_SAMPLES} for the full MMHS150K dataset"
        )
        critical_fail = True

    # Load splits and check membership
    splits = {}
    for name, p in split_files.items():
        with open(p, "r", encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip()]
        splits[name] = ids
        print(f"Split {name}: {len(ids)} ids")

    # Check splits are subsets of GT
    gt_ids = set(gt.keys())
    for name, ids in splits.items():
        missing = [i for i in ids if i not in gt_ids]
        if missing:
            print(f"[CRITICAL] {len(missing)} IDs in {name} not found in GT (example: {missing[:5]})")
            critical_fail = True

    if critical_fail:
        print("Critical issues with splits vs GT — aborting.")
        sys.exit(2)

    # Image coverage check
    def img_exists(tid: str) -> bool:
        return (img_dir / f"{tid}.jpg").exists()

    for name, ids in splits.items():
        if not ids:
            continue
        n = len(ids)
        found = sum(1 for tid in ids if img_exists(tid))
        frac = found / n
        print(f"Image coverage for {name}: {found}/{n} ({frac:.2%})")
        if frac < args.min_img_coverage:
            print(f"[CRITICAL] Image coverage for {name} below threshold {args.min_img_coverage}")
            critical_fail = True

    if critical_fail:
        print("Critical image coverage issues — aborting.")
        sys.exit(2)

    # Try opening a few images
    sample_ids = splits["val"][: args.sample_check_images] if splits["val"] else list(gt.keys())[: args.sample_check_images]
    for tid in sample_ids:
        p = img_dir / f"{tid}.jpg"
        try:
            with Image.open(p) as im:
                im.verify()
            print(f"OK: image opens -> {p}")
        except Exception as e:
            print(f"[WARN] Failed to open image {p}: {e}")
            ok = False

    # OCR check
    if ocr_consolidated.exists():
        ocr = load_json(ocr_consolidated)
        if ocr is None:
            print("[WARN] OCR consolidated JSON exists but failed to parse")
            ok = False
        else:
            with_text = sum(1 for v in ocr.values() if v and str(v).strip())
            total = len(ocr)
            print(f"OCR consolidated entries: {total}, with text: {with_text} ({with_text/total:.2%})")
    elif ocr_dir_new.exists() or ocr_dir_old.exists():
        print("OCR consolidated missing but per-image OCR dirs exist (img_txt/img_txt_new).")
    else:
        print("[WARN] No OCR data found (neither consolidated nor per-image dirs)")
        ok = False

    # Processed labels
    if processed_labels_file.exists():
        print(f"OK: processed labels found -> {processed_labels_file}")
    else:
        print(f"[WARN] processed_labels.json not found -> {processed_labels_file}. Will be auto-generated at runtime if needed.")

    # Final summary
    if critical_fail:
        print("Precheck FAILED (critical). Fix the issues above before running full training.")
        sys.exit(2)
    elif not ok:
        print("Precheck completed with warnings. Review messages above.")
        sys.exit(1)
    else:
        print("Precheck OK — dataset looks ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
