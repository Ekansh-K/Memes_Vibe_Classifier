"""
merge_vlm_captions.py
─────────────────────
Merges vlm_captions.json (Qwen3-VL output) into ocr_consolidated.json
to produce a single 100%-coverage text file for every image in MMHS150K.

Usage:
    python scripts/merge_vlm_captions.py \\
        --ocr  dataset/ocr_consolidated.json \\
        --vlm  vlm_captions.json \\
        --out  dataset/ocr_vlm_merged.json

Output format (same as ocr_consolidated.json):
    {tweet_id: "text string"}   -- OCR text if available, else VLM caption

"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Merge VLM captions into OCR file.")
    ap.add_argument("--ocr", required=True, type=Path,
                    help="Path to ocr_consolidated.json (tweet_id -> text string)")
    ap.add_argument("--vlm", required=True, type=Path,
                    help="Path to vlm_captions.json (tweet_id -> {caption, elapsed_s})")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output path for merged file")
    ap.add_argument("--prefer", choices=["ocr", "vlm"], default="ocr",
                    help="Which source to prefer when BOTH have text (default: ocr)")
    args = ap.parse_args()

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"Loading OCR file  : {args.ocr}")
    with open(args.ocr, encoding="utf-8") as f:
        ocr: dict[str, str] = json.load(f)

    print(f"Loading VLM file  : {args.vlm}")
    with open(args.vlm, encoding="utf-8") as f:
        vlm_raw: dict = json.load(f)

    # Normalise VLM to {tid: str}
    vlm: dict[str, str] = {}
    for tid, val in vlm_raw.items():
        if isinstance(val, dict):
            vlm[tid] = val.get("caption", "") or ""
        else:
            vlm[tid] = str(val) if val else ""

    # ── Merge ───────────────────────────────────────────────────────────────
    merged: dict[str, str] = {}
    ocr_used = vlm_used = both = 0

    all_ids = set(ocr) | set(vlm)
    for tid in all_ids:
        ocr_text = str(ocr.get(tid, "")).strip()
        vlm_text = str(vlm.get(tid, "")).strip()

        if ocr_text and vlm_text:
            both += 1
            merged[tid] = ocr_text if args.prefer == "ocr" else vlm_text
            ocr_used += args.prefer == "ocr"
            vlm_used += args.prefer == "vlm"
        elif ocr_text:
            merged[tid] = ocr_text
            ocr_used += 1
        elif vlm_text:
            merged[tid] = vlm_text
            vlm_used += 1
        else:
            merged[tid] = ""  # still empty (shouldn't happen for targeted IDs)

    # ── Stats ────────────────────────────────────────────────────────────────
    total   = len(merged)
    filled  = sum(1 for v in merged.values() if v)
    empty   = total - filled
    print(f"\nMerge stats:")
    print(f"  Total IDs    : {total:,}")
    print(f"  OCR only     : {ocr_used:,}")
    print(f"  VLM only     : {vlm_used:,}")
    print(f"  Both had text: {both:,}  (kept {'OCR' if args.prefer == 'ocr' else 'VLM'})")
    print(f"  Filled       : {filled:,}  ({100*filled/max(total,1):.1f}%)")
    print(f"  Still empty  : {empty:,}  ({100*empty/max(total,1):.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)
    size_mb = args.out.stat().st_size / 1e6
    print(f"\nSaved -> {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
