"""Test OCR on 5 images known to have text (from old OCR), verify output is saved."""
import os, json, sys
from pathlib import Path

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OLD_DIR = PROJECT_ROOT / "dataset" / "img_txt"
IMG_DIR = PROJECT_ROOT / "dataset" / "img_resized"
OUT_DIR = PROJECT_ROOT / "dataset" / "img_txt_new"
OUT_DIR.mkdir(exist_ok=True)

# Pick 5 images that have non-empty old OCR
test_ids = []
for p in OLD_DIR.glob("*.json"):
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("img_text", "").strip() and (IMG_DIR / p.name.replace(".json", ".jpg")).exists():
        test_ids.append(p.stem)
    if len(test_ids) == 5:
        break

print(f"Selected {len(test_ids)} test images with known old OCR text\n")

from scripts.preprocess_ocr import PaddleOCREngine
engine = PaddleOCREngine(device="gpu:0")

print("=" * 65)
print("  SAMPLE OCR COMPARISON: Old (2018) vs New (PP-OCRv5 GPU)")
print("=" * 65)

all_ok = True
for tid in test_ids:
    img_path = str(IMG_DIR / f"{tid}.jpg")
    old_text = json.loads((OLD_DIR / f"{tid}.json").read_text(encoding="utf-8")).get("img_text", "")

    result = engine.extract(img_path)

    # Save to JSON
    out_path = OUT_DIR / f"{tid}.json"
    result["tweet_id"] = tid
    result["ocr_engine"] = engine.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    saved = json.loads(out_path.read_text(encoding="utf-8"))

    print(f"\nID : {tid}")
    print(f"  OLD : {repr(old_text[:120])}")
    print(f"  NEW : {repr(result['ocr_text'][:120])}")
    print(f"  conf={result['confidence']:.2f}  boxes={result['num_boxes']}  saved_ok={'✅' if saved['ocr_text']==result['ocr_text'] else '❌'}")

    if saved["ocr_text"] != result["ocr_text"]:
        all_ok = False

print("\n" + "=" * 65)
print(f"  Saved {len(test_ids)} files to dataset/img_txt_new/")
print(f"  All files verified: {'✅ PASS' if all_ok else '❌ MISMATCH'}")
