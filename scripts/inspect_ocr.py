import json, os, random
from pathlib import Path
from collections import Counter

ocr_dir = Path("dataset/img_txt")
gt_file = Path("dataset/MMHS150K_GT.json")

with open(gt_file) as f:
    gt = json.load(f)

ocr_files = list(ocr_dir.glob("*.json"))
print(f"Total OCR files: {len(ocr_files)}")
print(f"Total GT entries: {len(gt)}")
print(f"Missing OCR: {len(gt) - len(ocr_files)} ({100*(len(gt)-len(ocr_files))/len(gt):.1f}%)")

# Inspect structure of first OCR file
print("\n--- Sample OCR file structures ---")
for f in ocr_files[:3]:
    with open(f) as fp:
        data = json.load(fp)
    print(f"\nFile: {f.name}")
    print(f"  Type: {type(data)}")
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
    elif isinstance(data, list):
        print(f"  Length: {len(data)}, first item: {data[0] if data else 'EMPTY'}")
    print(f"  Raw: {str(data)[:200]}")

# Now analyze content quality across all OCR files
random.seed(42)
sample_ids = random.sample(ocr_files, min(2000, len(ocr_files)))

empty = 0
very_short = 0  # < 5 chars
short = 0       # 5-20 chars
medium = 0      # 20-100 chars
long_ = 0       # > 100 chars
noisy_chars = 0 # high non-ASCII ratio

texts = []
for f in sample_ids:
    with open(f) as fp:
        data = json.load(fp)
    
    # Extract text from whatever structure
    if isinstance(data, list):
        text = " ".join(str(x) for x in data if x)
    elif isinstance(data, dict):
        text = " ".join(str(v) for v in data.values() if v)
    elif isinstance(data, str):
        text = data
    else:
        text = str(data)
    
    text = text.strip()
    texts.append((f.stem, text))
    
    if not text:
        empty += 1
    elif len(text) < 5:
        very_short += 1
    elif len(text) < 20:
        short += 1
    elif len(text) < 100:
        medium += 1
    else:
        long_ += 1
    
    # Check for noisy characters
    if text:
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.2:
            noisy_chars += 1

n = len(sample_ids)
print(f"\n--- OCR Content Quality (sample of {n}) ---")
print(f"  Empty/null:           {empty:5d} ({100*empty/n:.1f}%)")
print(f"  Very short (<5 chars): {very_short:5d} ({100*very_short/n:.1f}%)")
print(f"  Short (5-20 chars):   {short:5d} ({100*short/n:.1f}%)")
print(f"  Medium (20-100 chars): {medium:5d} ({100*medium/n:.1f}%)")
print(f"  Long (>100 chars):    {long_:5d} ({100*long_/n:.1f}%)")
print(f"  Noisy (>20% non-ASCII):{noisy_chars:5d} ({100*noisy_chars/n:.1f}%)")

# Show examples of different quality levels
print("\n--- Example OCR outputs ---")
noisy_examples = [(tid, t) for tid, t in texts if t and sum(1 for c in t if ord(c)>127)/len(t)>0.2][:3]
empty_examples = [(tid, t) for tid, t in texts if not t][:3]
good_examples = [(tid, t) for tid, t in texts if 20 < len(t) < 200][:5]

print("\nGOOD OCR examples:")
for tid, t in good_examples:
    tweet = gt.get(tid, {}).get("tweet_text", "N/A")
    print(f"  Tweet: {tweet[:70]}")
    print(f"  OCR:   {t[:120]}")
    print()

print("EMPTY OCR (text exists in GT but extracted nothing):")
for tid, t in empty_examples[:3]:
    tweet = gt.get(tid, {}).get("tweet_text", "N/A")
    labels = gt.get(tid, {}).get("labels", [])
    print(f"  Tweet: {tweet[:70]}")
    print(f"  Labels: {labels}")
    print()

print("NOISY/HIGH NON-ASCII OCR:")
for tid, t in noisy_examples[:3]:
    tweet = gt.get(tid, {}).get("tweet_text", "N/A")
    print(f"  Tweet: {tweet[:70]}")
    print(f"  OCR:   {t[:120]}")
    print()

# Cross-check: samples that have OCR but OCR text matches tweet text closely
print("\n--- OCR text vs Tweet text overlap ---")
overlap_count = 0
total_with_ocr_text = 0
for tid, t in texts:
    if not t:
        continue
    total_with_ocr_text += 1
    tweet_words = set(gt.get(tid, {}).get("tweet_text", "").lower().split())
    ocr_words = set(t.lower().split())
    if tweet_words and ocr_words:
        overlap = len(tweet_words & ocr_words) / max(len(tweet_words), 1)
        if overlap > 0.5:
            overlap_count += 1

print(f"  OCR files with non-empty text: {total_with_ocr_text}/{n} ({100*total_with_ocr_text/n:.1f}%)")
print(f"  OCR text overlaps >50% with tweet text: {overlap_count}/{total_with_ocr_text} ({100*overlap_count/max(total_with_ocr_text,1):.1f}%)")
print("  (High overlap = OCR is extracting tweet text, not image text = redundant!)")
