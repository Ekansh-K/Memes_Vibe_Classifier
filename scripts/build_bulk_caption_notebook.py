"""
Generates the bulk VLM captioning notebook for Kaggle.

This notebook:
  1. Reads ocr_consolidated.json (uploaded as a second Kaggle input dataset)
  2. Identifies the ~67K images that still have no OCR text (value == "")
  3. Runs Qwen3-VL-8B-Instruct FP16 on those images only
  4. Saves vlm_captions.json -> /kaggle/working/vlm_captions.json

API: confirmed working from vram_test_corrected.ipynb (Kaggle dual-T4 run)
  - Class: AutoModelForImageTextToText
  - process_vision_info: 3-value unpack with return_video_kwargs=True
  - Device: next(model.parameters()).device
  - Processor: min_pixels=256*256, max_pixels=448*448
"""
import json
from pathlib import Path

BASE_DS  = "/kaggle/input/datasets/victorcallejasf/multimodal-hate-speech"
# ocr_consolidated.json is uploaded as a second Kaggle input dataset
# Adjust the slug if you rename it when uploading
OCR_DS   = "/kaggle/input/mmhs150k-ocr-consolidated"

OUT_NB   = Path(__file__).parent.parent / "notebooks" / "vlm_bulk_caption_kaggle.ipynb"

CELLS_RAW = [

# 0 ── markdown ────────────────────────────────────────────────────────────────
("markdown", """\
# MMHS150K — Bulk VLM Captioning (Qwen3-VL-8B FP16)

**Goal**: Generate image descriptions for the ~67K images that have no OCR text
after running PaddleOCR locally (55% coverage → 100% coverage).

**Inputs**:
- `victorcallejasf/multimodal-hate-speech` — base dataset (images)
- `mmhs150k-ocr-consolidated` — your locally-improved `ocr_consolidated.json`
  (upload this as a Kaggle dataset before running)

**Output**: `/kaggle/working/vlm_captions.json`
  Format: `{tweet_id: {"caption": "...", "elapsed_s": 4.8}}`

**Model**: Qwen3-VL-8B-Instruct FP16, `device_map='auto'` (dual T4, ~17GB)
**API**: confirmed working from `vram_test_corrected.ipynb`
"""),

# 1 ── install ─────────────────────────────────────────────────────────────────
("code", """\
!pip install -q 'transformers>=4.57.0' bitsandbytes accelerate 'qwen_vl_utils>=0.0.14' Pillow tqdm
print('done')
"""),

# 2 ── config ──────────────────────────────────────────────────────────────────
("code", f"""\
import json, gc, time, os
from pathlib import Path

# ── dataset paths ─────────────────────────────────────────────────────────────
BASE_DS   = Path('{BASE_DS}')
OCR_DS    = Path('{OCR_DS}')
WORK_DIR  = Path('/kaggle/working')

IMG_DIR   = BASE_DS / 'img_resized'
OCR_JSON  = OCR_DS  / 'ocr_consolidated.json'   # your locally-improved file

OUT_FILE  = WORK_DIR / 'vlm_captions.json'
PROG_FILE = WORK_DIR / 'vlm_captions_progress.json'  # checkpoint every N images

# ── model ─────────────────────────────────────────────────────────────────────
MODEL_ID  = 'Qwen/Qwen3-VL-8B-Instruct'

# ── prompt (single, optimised for hate-speech context) ────────────────────────
PROMPT = (
    'Describe this social media image briefly and factually in 1-2 sentences. '
    'Note any visible text, people, symbols, logos, memes, or potentially '
    'harmful/offensive content. Be objective.'
)

# ── batching ─────────────────────────────────────────────────────────────────
BATCH_SIZE       = 1      # Qwen3-VL works best with batch=1 for mixed-size images
SAVE_EVERY       = 500    # checkpoint every 500 images
MAX_NEW_TOKENS   = 120    # caption length cap

# ── verify paths ─────────────────────────────────────────────────────────────
for p in [IMG_DIR, OCR_JSON]:
    print(f"  {{'OK' if p.exists() else 'MISSING'}}: {{p}}")
"""),

# 3 ── load OCR, find missing IDs ─────────────────────────────────────────────
("code", """\
print('Loading ocr_consolidated.json ...')
with open(OCR_JSON, encoding='utf-8') as f:
    ocr = json.load(f)

total      = len(ocr)
have_text  = {tid for tid, txt in ocr.items() if str(txt).strip()}
need_text  = {tid for tid, txt in ocr.items() if not str(txt).strip()}

# only keep IDs whose image actually exists on disk
need_text = {tid for tid in need_text if (IMG_DIR / f'{tid}.jpg').exists()}

print(f'  Total entries    : {total:,}')
print(f'  Already have OCR : {len(have_text):,}  ({100*len(have_text)/total:.1f}%)')
print(f'  Need VLM caption : {len(need_text):,}  ({100*len(need_text)/total:.1f}%)')

# Convert to a sorted list so runs are deterministic / resumable
need_ids = sorted(need_text)
print(f'  Image files found: {len(need_ids):,}')
"""),

# 4 ── resume from checkpoint if exists ───────────────────────────────────────
("code", """\
# Load existing progress so interrupted runs can resume cleanly
if PROG_FILE.exists():
    with open(PROG_FILE, encoding='utf-8') as f:
        done_captions = json.load(f)
    print(f'Resuming: {len(done_captions):,} captions already done')
else:
    done_captions = {}
    print('Starting fresh (no checkpoint found)')

# Filter out already-done IDs
remaining_ids = [tid for tid in need_ids if tid not in done_captions]
print(f'Remaining to caption: {len(remaining_ids):,}')

# Estimate runtime
est_hours = len(remaining_ids) * 4.8 / 3600
print(f'Estimated time at ~4.8s/image: {est_hours:.1f} hours')
print(f'  (Kaggle session limit: 12 h — will checkpoint every {SAVE_EVERY} images)')
"""),

# 5 ── load model ─────────────────────────────────────────────────────────────
("code", """\
import torch
from PIL import Image
from tqdm.notebook import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

print(f'Loading {MODEL_ID} FP16 (device_map=auto) ...')
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 256,
    max_pixels=448 * 448,
)

for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    total_vram = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU{i}: {alloc:.1f} / {total_vram:.1f} GB used')

DEVICE = next(model.parameters()).device
print(f'Primary device: {DEVICE}')
"""),

# 6 ── bulk inference loop ─────────────────────────────────────────────────────
("code", """\
def caption_image(img_path: str) -> tuple[str, float]:
    \"\"\"Run Qwen3-VL on a single image, return (caption, elapsed_s).\"\"\"
    img = Image.open(img_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text',  'text':  PROMPT},
    ]}]
    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 3-value unpack — confirmed working in vram_test_corrected.ipynb
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text_in],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors='pt',
        **video_kwargs,
    ).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0

    prompt_len = inputs['input_ids'].shape[1]
    caption = processor.decode(
        out_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()
    return caption, round(elapsed, 3)


# ── main loop ────────────────────────────────────────────────────────────────
errors = {}
batch_times = []

print(f'Starting bulk captioning: {len(remaining_ids):,} images')
print(f'Checkpoint every {SAVE_EVERY} images -> {PROG_FILE.name}')
print('─' * 60)

for idx, tid in enumerate(tqdm(remaining_ids, desc='VLM captioning')):
    img_path = IMG_DIR / f'{tid}.jpg'
    try:
        caption, elapsed = caption_image(str(img_path))
        done_captions[tid] = {'caption': caption, 'elapsed_s': elapsed}
        batch_times.append(elapsed)
    except Exception as e:
        errors[tid] = str(e)
        done_captions[tid] = {'caption': '', 'elapsed_s': 0.0}

    # checkpoint save
    if (idx + 1) % SAVE_EVERY == 0:
        with open(PROG_FILE, 'w', encoding='utf-8') as f:
            json.dump(done_captions, f, ensure_ascii=False)
        avg = sum(batch_times[-SAVE_EVERY:]) / len(batch_times[-SAVE_EVERY:])
        remaining = len(remaining_ids) - (idx + 1)
        eta_h = remaining * avg / 3600
        print(f'  [{idx+1:>6}/{len(remaining_ids)}] checkpoint saved | '
              f'avg={avg:.2f}s | ETA~{eta_h:.1f}h | errors={len(errors)}')

# final save
with open(PROG_FILE, 'w', encoding='utf-8') as f:
    json.dump(done_captions, f, ensure_ascii=False)

print(f'\\nDone. {len(done_captions):,} captions | {len(errors):,} errors')
if batch_times:
    print(f'Avg inference: {sum(batch_times)/len(batch_times):.2f}s/image')
"""),

# 7 ── write final output ──────────────────────────────────────────────────────
("code", """\
# Write the final clean output file
# Format: {tweet_id: {"caption": "...", "elapsed_s": 4.8}}
with open(OUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(done_captions, f, indent=2, ensure_ascii=False)

# Stats
have_cap  = sum(1 for v in done_captions.values() if v['caption'])
empty_cap = sum(1 for v in done_captions.values() if not v['caption'])
print(f'vlm_captions.json written -> {OUT_FILE}')
print(f'  Total  : {len(done_captions):,}')
print(f'  Filled : {have_cap:,}  ({100*have_cap/max(len(done_captions),1):.1f}%)')
print(f'  Empty  : {empty_cap:,}  ({100*empty_cap/max(len(done_captions),1):.1f}%)')
print()
print('Files in /kaggle/working/:')
for f in sorted(WORK_DIR.iterdir()):
    sz_mb = f.stat().st_size / 1e6
    print(f'  {f.name:<45s}  {sz_mb:>7.1f} MB')
print('\\nDownload vlm_captions.json from the Kaggle output tab.')
"""),

# 8 ── merge locally (info cell) ───────────────────────────────────────────────
("markdown", """\
## After downloading `vlm_captions.json`

Run the local merge script to produce the final 100%-coverage file:

```bash
python scripts/merge_vlm_captions.py \\
    --ocr   dataset/ocr_consolidated.json \\
    --vlm   vlm_captions.json \\
    --out   dataset/ocr_vlm_merged.json
```

The merged file has the same `{tweet_id: text}` format as `ocr_consolidated.json`
so all downstream code (splits, training) works unchanged.

### Coverage after merge
| Source | Images | % |
|---|---|---|
| PaddleOCR (local) | ~82,479 | 55.0% |
| Qwen3-VL captions | ~67,521 | 45.0% |
| **Total** | **150,000** | **100%** |
"""),

]  # end CELLS_RAW


# ── notebook builder ──────────────────────────────────────────────────────────
def make_cell(cell_type, source):
    lines = [ln + "\n" for ln in source.splitlines()]
    while lines and lines[-1] == "\n":
        lines.pop()
    cell = {"cell_type": cell_type, "metadata": {}, "source": lines}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


nb = {
    "cells": [make_cell(ct, src) for ct, src in CELLS_RAW],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.12",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_NB.parent.mkdir(parents=True, exist_ok=True)
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {OUT_NB}")
