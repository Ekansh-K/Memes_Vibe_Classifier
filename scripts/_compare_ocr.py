"""Quick OCR quality comparison: old vs new on overlap samples."""
import json, random
from pathlib import Path

NEW_DIR = Path("dataset/img_txt_new")
OLD_DIR = Path("dataset/img_txt")

# Get all new files that have non-empty text
new_files = list(NEW_DIR.glob("*.json"))
print(f"New OCR files on disk: {len(new_files):,}")

non_empty, overlap_with_text, empty_new = [], [], 0
for p in new_files:
    with open(p, encoding="utf-8") as f:
        nd = json.load(f)
    txt = nd.get("ocr_text", "").strip()
    if txt:
        non_empty.append((p.stem, txt, nd.get("confidence", 0), nd.get("num_boxes", 0)))
    else:
        empty_new += 1

print(f"Non-empty OCR (text found): {len(non_empty):,}  ({len(non_empty)/len(new_files)*100:.1f}%)")
print(f"Empty (no text in image):   {empty_new:,}  ({empty_new/len(new_files)*100:.1f}%)\n")

# Find overlap with old OCR for quality comparison
random.seed(42)
random.shuffle(non_empty)
compared = []
for stem, new_txt, conf, boxes in non_empty:
    old_p = OLD_DIR / f"{stem}.json"
    if not old_p.exists():
        continue
    with open(old_p, encoding="utf-8") as f:
        old_txt = json.load(f).get("img_text", "").strip()
    compared.append((stem, old_txt, new_txt, conf, boxes))
    if len(compared) >= 10:
        break

print(f"{'='*70}")
print(f"  QUALITY COMPARISON: Old (2018) vs New (PP-OCRv5 GPU)")
print(f"{'='*70}")
for tid, old_t, new_t, conf, boxes in compared:
    print(f"\nTweet {tid}:")
    print(f"  OLD : {(old_t[:150] if old_t else '(empty)')!r}")
    print(f"  NEW : {new_t[:150]!r}")
    print(f"        conf={conf:.2f}, boxes={boxes}")
