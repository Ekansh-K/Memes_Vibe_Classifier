"""A1 Fix: Merge missing img_txt_new entries into ocr_consolidated.json.

All 3,087 missing IDs were fully processed by PaddleOCR and have JSON files
in dataset/img_txt_new/ — they just never got written to the consolidated file
(likely due to the periodic-save logic triggering at exact multiples of 1000).

This script:
1. Finds all GT IDs missing from ocr_consolidated.json
2. Reads their per-image JSONs from img_txt_new/
3. Merges them into ocr_consolidated.json (in-place, atomic write)
4. Reports a summary of what changed

Usage:
    python scripts/fix_consolidated_ocr.py
    python scripts/fix_consolidated_ocr.py --dry-run   # preview only, no changes
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import GT_FILE, OCR_CONSOLIDATED, OCR_DIR_NEW


def main():
    parser = argparse.ArgumentParser(description="Fix missing entries in ocr_consolidated.json")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, make no changes")
    args = parser.parse_args()

    # ── Load existing data ──────────────────────────────────────────────────
    print(f"[INFO] Loading GT JSON from {GT_FILE} ...")
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gt_ids = set(gt.keys())
    print(f"[INFO] GT entries: {len(gt_ids):,}")

    print(f"[INFO] Loading consolidated OCR from {OCR_CONSOLIDATED} ...")
    with open(OCR_CONSOLIDATED, "r", encoding="utf-8") as f:
        consolidated = json.load(f)
    print(f"[INFO] Consolidated entries before fix: {len(consolidated):,}")

    # ── Find missing IDs ────────────────────────────────────────────────────
    missing_ids = sorted(gt_ids - set(consolidated.keys()))
    print(f"[INFO] Missing from consolidated: {len(missing_ids):,}")

    if not missing_ids:
        print("[INFO] Nothing to fix - consolidated is already complete!")
        return

    # ── Read per-image JSONs and merge ──────────────────────────────────────
    found_in_new = []
    empty_text = []
    has_text = []
    not_found = []

    for tid in missing_ids:
        p = OCR_DIR_NEW / f"{tid}.json"
        if not p.exists():
            not_found.append(tid)
            consolidated[tid] = ""
            continue

        # Use context manager to ensure the file handle is always closed
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Failed to read {p}: {e}")
            not_found.append(tid)
            consolidated[tid] = ""
            continue

        ocr_text = data.get("ocr_text", "").strip()
        consolidated[tid] = ocr_text
        found_in_new.append(tid)

        if ocr_text:
            has_text.append(tid)
        else:
            empty_text.append(tid)

    # ── Report ──────────────────────────────────────────────────────────────
    print("\n[SUMMARY]")
    print(f"  Found in img_txt_new/:  {len(found_in_new):,}")
    print(f"    -> With text:         {len(has_text):,}")
    print(f"    -> Empty (no text):   {len(empty_text):,}")
    print(f"  Not found anywhere:     {len(not_found):,}")
    print(f"  New consolidated size:  {len(consolidated):,}")

    if not_found:
        print(f"\n  [WARN] These {len(not_found)} IDs were not in img_txt_new/:")
        for tid in not_found[:10]:
            print(f"    {tid}")
        if len(not_found) > 10:
            print(f"    ... and {len(not_found) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # ── Atomic write (write to .tmp then rename to avoid corruption) ────────
    tmp_path = OCR_CONSOLIDATED.with_suffix(".json.tmp")
    print(f"[INFO] Writing updated consolidated OCR to {OCR_CONSOLIDATED} ...")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False)
    tmp_path.replace(OCR_CONSOLIDATED)
    print(f"[INFO] Done. ocr_consolidated.json now has {len(consolidated):,} entries.")

    # ── Quick sanity check ──────────────────────────────────────────────────
    still_missing = gt_ids - set(consolidated.keys())
    if still_missing:
        print(f"[WARN] Still missing after fix: {len(still_missing):,} IDs")
    else:
        print(f"[OK] All {len(gt_ids):,} GT IDs are now in ocr_consolidated.json (complete)")


if __name__ == "__main__":
    main()
