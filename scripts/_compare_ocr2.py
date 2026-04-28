import json, os
from pathlib import Path

new_dir = Path("dataset/img_txt_new")
old_dir = Path("dataset/img_txt")

files = os.listdir(new_dir)[:500]
results = []
for fname in files:
    nd = json.loads((new_dir / fname).read_text(encoding="utf-8"))
    txt = nd.get("ocr_text", "").strip()
    if not txt:
        continue
    old_p = old_dir / fname
    old_txt = ""
    if old_p.exists():
        old_txt = json.loads(old_p.read_text(encoding="utf-8")).get("img_text", "").strip()
    results.append((fname[:-5], old_txt, txt, nd.get("confidence", 0), nd.get("num_boxes", 0)))
    if len(results) >= 10:
        break

# Also get overall coverage stats from first 500
total = len(files)
non_empty = sum(1 for f in files if json.loads((new_dir/f).read_text(encoding="utf-8")).get("ocr_text","").strip())
print(f"Sample coverage ({total} files): {non_empty} have text ({non_empty/total*100:.1f}%)\n")

print("=" * 68)
print("  OLD (2018 OCR)  vs  NEW (PP-OCRv5 GPU)")
print("=" * 68)
for tid, ol, nw, c, b in results:
    print(f"\nID: {tid}")
    old_display = repr(ol[:130]) if ol else "(empty - no old OCR)"
    print(f"  OLD: {old_display}")
    print(f"  NEW: {repr(nw[:130])}")
    print(f"       conf={c:.2f}, boxes={b}")
