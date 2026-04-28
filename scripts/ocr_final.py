import json
from pathlib import Path
from collections import Counter

ocr_dir = Path("dataset/img_txt")
gt_file = Path("dataset/MMHS150K_GT.json")

with open(gt_file) as f:
    gt = json.load(f)

short_and_noisy = 0
screenshot_like = 0
url_count = 0
total = 0

for f in ocr_dir.glob("*.json"):
    with open(f) as fp:
        d = json.load(fp)
    t = d.get("img_text", "").strip()
    total += 1
    if len(t) < 10:
        short_and_noisy += 1
    if any(x in t for x in ["LTE", "Verizon", "AT&T", "% ", "iPhone", "Android", "Wi-Fi", "WIFI"]):
        screenshot_like += 1

for tid, entry in gt.items():
    if "https://t.co" in entry.get("tweet_text", "") or "http://" in entry.get("tweet_text", ""):
        url_count += 1

print(f"OCR < 10 chars (basically useless): {short_and_noisy} ({100*short_and_noisy/total:.1f}%)")
print(f"Screenshot UI noise in OCR:         {screenshot_like} ({100*screenshot_like/total:.1f}%)")
print(f"GT entries with t.co URLs:          {url_count} ({100*url_count/len(gt):.1f}%)")
print(f"  (content is inside the image, tweet text is just a link)")
