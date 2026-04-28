"""
Builds vlm_caption_eval_kaggle.ipynb.

What's in this notebook:
  - A2 (re-run): Complete OCR noise filter on full 150K ocr_consolidated.json
    (A1 is already done locally; uploading dataset includes all 150K OCR files)
  - A3: Qwen3-VL-8B FP16 × 3 prompts × 200-image stratified eval sample
"""
import json
from pathlib import Path

DATASET_PATH = "/kaggle/input/mmhs150k-full"   # ← update slug to match your upload
OUT_NB = Path(__file__).parent.parent / "notebooks" / "vlm_caption_eval_kaggle.ipynb"

CELLS_RAW = [

# ── 0: markdown ──────────────────────────────────────────────────────────────
("markdown", """\
# MMHS150K — A2 + A3 Pipeline (Qwen3-VL-8B FP16)

**What runs here**
- **A2** Re-apply OCR noise filter to the full 150K `ocr_consolidated.json`
  → produces `ocr_filtered.json` in `/kaggle/working/`
- **A3** VLM Caption Evaluation — Qwen3-VL-8B-Instruct FP16 × 3 prompts × 200 images

> ⚠️ Set `CODEBASE_DIR` in the paths cell to wherever you placed the repo in `/kaggle/working/`.
"""),

# ── 1: install ────────────────────────────────────────────────────────────────
("code", """\
!pip install -q 'transformers>=4.57.0' bitsandbytes accelerate 'qwen_vl_utils>=0.0.14' Pillow tqdm
print('done')
"""),

# ── 2: paths ─────────────────────────────────────────────────────────────────
("code", f"""\
import json, gc, sys, time
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# ADJUSTABLE: path to your codebase inside /kaggle/working/
# e.g. if you unzipped EndSem_Project.zip → /kaggle/working/EndSem_Project
CODEBASE_DIR = Path('/kaggle/working/EndSem_Project')
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(CODEBASE_DIR))

INPUT_DIR  = Path('{DATASET_PATH}')
WORK_DIR   = Path('/kaggle/working')
GT_FILE    = INPUT_DIR / 'MMHS150K_GT.json'
IMG_TXT    = INPUT_DIR / 'img_txt'
IMG_DIR    = INPUT_DIR / 'img_resized'
SPLITS_DIR = INPUT_DIR / 'splits'
OCR_CONS   = INPUT_DIR / 'ocr_consolidated.json'
EVAL_DIR   = WORK_DIR  / 'vlm_eval'
EVAL_DIR.mkdir(parents=True, exist_ok=True)

for p in [GT_FILE, IMG_TXT, IMG_DIR, SPLITS_DIR, OCR_CONS, CODEBASE_DIR]:
    status = 'OK' if p.exists() else 'MISSING'
    print(f'  [{{status}}] {{p}}')
"""),

# ── 3: A2 — apply OCR noise filter to full 150K consolidated ─────────────────
("code", """\
# ── A2: Re-apply OCR noise filter to ocr_consolidated.json (150K entries)
# A1 is already complete — all images are OCR'd in the uploaded dataset.
# A2 needs a full re-run because ocr_consolidated_filtered.json in the dataset
# was built before A1 completed (only covered 146,913 / 150,000 entries).

OCR_FILTERED_PATH = WORK_DIR / 'ocr_filtered.json'

if OCR_FILTERED_PATH.exists():
    with open(OCR_FILTERED_PATH, encoding='utf-8') as f:
        ocr_filtered = json.load(f)
    print(f'[A2] Loaded existing ocr_filtered.json ({len(ocr_filtered):,} entries)')
else:
    # Load the full consolidated OCR
    print(f'[A2] Loading {OCR_CONS} ...')
    with open(OCR_CONS, encoding='utf-8') as f:
        ocr_data = json.load(f)
    print(f'[A2] Loaded {len(ocr_data):,} entries')

    # Import noise filter from codebase
    try:
        from src.data.ocr_filter import filter_ocr_noise
        print('[A2] Imported filter_ocr_noise from src/data/ocr_filter.py')
    except ImportError as e:
        print(f'[WARN] Could not import ocr_filter ({e}) — using identity filter')
        def filter_ocr_noise(t): return t

    print('[A2] Applying noise filter ...')
    ocr_filtered = {}
    noise_count  = 0
    for tid, raw in ocr_data.items():
        cleaned = filter_ocr_noise(raw) if raw else ''
        ocr_filtered[tid] = cleaned
        if cleaned != raw:
            noise_count += 1

    with open(OCR_FILTERED_PATH, 'w', encoding='utf-8') as f:
        json.dump(ocr_filtered, f, ensure_ascii=False)

    with_text = sum(1 for v in ocr_filtered.values() if v and v.strip())
    print(f'[A2] Done. {noise_count:,} entries had noise removed.')
    print(f'[A2] Entries with text after filter: {with_text:,} / {len(ocr_filtered):,}')
    print(f'[A2] Saved → {OCR_FILTERED_PATH}')
"""),

# ── 4: label + OCR helpers ────────────────────────────────────────────────────
("code", """\
def gt_binary_label(entry):
    \"\"\"Majority-vote binary label. 0=NotHate, 1=Hate.\"\"\"
    raw = entry.get('labels', [])
    if not raw:
        return 0
    majority = Counter(raw).most_common(1)[0][0]
    return 0 if majority == 0 else 1


def get_ocr_text(tweet_id):
    \"\"\"Return filtered OCR text for tweet_id.\"\"\"
    return ocr_filtered.get(tweet_id, '') or ''


print('Helpers defined.')
"""),

# ── 5: build stratified 200-image eval sample ─────────────────────────────────
("code", """\
import random
random.seed(42)

with open(GT_FILE, encoding='utf-8') as f:
    gt = json.load(f)
with open(SPLITS_DIR / 'test_ids.txt', encoding='utf-8') as f:
    test_ids = set(f.read().strip().split('\\n'))

SAMPLE_PATH = EVAL_DIR / 'eval_sample.json'

if SAMPLE_PATH.exists():
    with open(SAMPLE_PATH, encoding='utf-8') as f:
        sample = json.load(f)
    print(f'Loaded existing sample: {len(sample)} items')
else:
    strata_names = {
        (True,  True):  'has_ocr_hate',
        (True,  False): 'has_ocr_nothate',
        (False, True):  'no_ocr_hate',
        (False, False): 'no_ocr_nothate',
    }
    candidates = {k: [] for k in strata_names}
    for tid in test_ids:
        if tid not in gt:
            continue
        if not (IMG_DIR / f'{tid}.jpg').exists():
            continue
        ocr = get_ocr_text(tid)
        candidates[(bool(ocr), gt_binary_label(gt[tid]) == 1)].append(tid)

    sample = []
    for key, name in strata_names.items():
        pool = candidates[key]
        if not pool:
            print(f'  [WARN] Stratum {name!r} is empty!')
            continue
        chosen = random.sample(pool, min(50, len(pool)))
        for tid in chosen:
            sample.append({
                'tweet_id':          tid,
                'stratum':           name,
                'tweet_text':        gt[tid].get('tweet_text', ''),
                'ocr_text':          get_ocr_text(tid),
                'hard_label_binary': gt_binary_label(gt[tid]),
                'img_path':          str(IMG_DIR / f'{tid}.jpg'),
            })
        print(f'  {name}: {len(chosen)} chosen / {len(pool)} available')

    with open(SAMPLE_PATH, 'w', encoding='utf-8') as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f'Total: {len(sample)} items | Saved -> {SAMPLE_PATH}')
"""),

# ── 6: shared inference helpers ───────────────────────────────────────────────
("code", """\
import torch
from PIL import Image
from tqdm.notebook import tqdm

PROMPTS = {
    'prompt_1': (
        'Describe this image in one or two detailed sentences. '
        'Focus on the people, objects, text, symbols, and overall scene.'
    ),
    'prompt_2': (
        'This image was posted on social media. Describe in detail:\\n'
        '1. Any people visible and what they are doing\\n'
        '2. Any text, signs, symbols, or logos visible in the image\\n'
        '3. Any offensive, hateful, or sensitive imagery (if present)\\n'
        '4. The overall tone/mood of the image\\n\\n'
        'Be factual and objective. Two to three sentences maximum.'
    ),
    'prompt_3': (
        'Analyze this social media image for content moderation purposes. '
        'Provide a brief, factual description covering:\\n'
        '- Visual content (people, objects, setting)\\n'
        '- Any embedded text, memes, or symbols\\n'
        '- Whether the image contains potentially harmful stereotypes, '
        'slurs, violent imagery, or hate symbols\\n'
        'Answer in 2-3 sentences. Be objective.'
    ),
}


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f'[VRAM freed] Allocated: {alloc:.2f} GB')


def save_results(model_tag, prompt_name, captions):
    out = EVAL_DIR / f'{model_tag}_{prompt_name}.json'
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(captions, fh, indent=2, ensure_ascii=False)
    times = [v['elapsed_s'] for v in captions.values()]
    lens  = [len(v['caption']) for v in captions.values() if v['caption']]
    empty = sum(1 for v in captions.values() if not v['caption'])
    avg_t = sum(times) / max(len(times), 1)
    avg_l = sum(lens) // max(len(lens), 1)
    print(f'  Saved {out.name} | avg={avg_t:.2f}s/img | avg_len={avg_l}c | empty={empty}')


print('Helpers loaded.')
"""),

# ── 7: A3 Qwen3-VL-8B FP16 inference ─────────────────────────────────────────
("code", """\
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# A3 — Qwen3-VL-8B-Instruct FP16  device_map='auto' (both T4 GPUs)
# CONFIRMED WORKING API (vram_test_corrected.ipynb):
#   Class : AutoModelForImageTextToText
#   Unpack: image_inputs, video_inputs, video_kwargs = process_vision_info(
#               messages, return_video_kwargs=True)   ← 3 values NOT 2
#   VRAM  : ~8.9 GB GPU0 + 8.8 GB GPU1 = 17.7 GB  |  4.8 s/image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_TAG = 'qwen3_8b_fp16'
MODEL_ID  = 'Qwen/Qwen3-VL-8B-Instruct'
already_done = all((EVAL_DIR / f'{MODEL_TAG}_{p}.json').exists() for p in PROMPTS)
model = processor = None

if already_done:
    print(f'[SKIP] {MODEL_TAG} — all result files exist already')
else:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f'Loading {MODEL_ID} FP16 ...')
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map='auto',
    ).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 256, max_pixels=448 * 448,
    )
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f'  GPU{i}: {alloc:.1f} / {total:.1f} GB')

    for prompt_name, prompt_text in PROMPTS.items():
        captions = {}
        for item in tqdm(sample, desc=f'{MODEL_TAG}/{prompt_name}'):
            img = Image.open(item['img_path']).convert('RGB')
            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text',  'text':  prompt_text},
            ]}]
            text_in = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = processor(
                text=[text_in], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors='pt', **video_kwargs,
            ).to(next(model.parameters()).device)

            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=150)
            elapsed = time.time() - t0

            prompt_len = inputs['input_ids'].shape[1]
            caption = processor.decode(
                out_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()
            captions[item['tweet_id']] = {
                'caption': caption, 'elapsed_s': round(elapsed, 3),
            }
        save_results(MODEL_TAG, prompt_name, captions)

    del model, processor
    free_vram()
"""),

# ── 8: summary table ──────────────────────────────────────────────────────────
("code", """\
import pandas as pd

rows = []
for result_file in sorted(EVAL_DIR.glob('qwen3_8b_fp16_prompt_*.json')):
    parts = result_file.stem.split('_prompt_')
    if len(parts) != 2:
        continue
    with open(result_file, encoding='utf-8') as fh:
        caps = json.load(fh)
    times = [v['elapsed_s'] for v in caps.values()]
    lens  = [len(v['caption']) for v in caps.values() if v['caption']]
    empty = sum(1 for v in caps.values() if not v['caption'])
    rows.append({
        'model':         parts[0],
        'prompt':        f'prompt_{parts[1]}',
        'n':             len(caps),
        'avg_time_s':    round(sum(times) / max(len(times), 1), 2),
        'avg_len_chars': round(sum(lens)  / max(len(lens),  1), 0),
        'empty_pct':     round(empty / max(len(caps), 1) * 100, 1),
    })

if rows:
    df = pd.DataFrame(rows)
    with open(EVAL_DIR / 'eval_summary.json', 'w', encoding='utf-8') as fh:
        df.to_json(fh, orient='records', indent=2, force_ascii=False)
    print('=== VLM Evaluation Summary ===')
    print(df.to_string(index=False))
else:
    print('[WARN] No result files found yet.')

if 'sample' in dir() and sample:
    target = next((s for s in sample if s['stratum'] == 'no_ocr_hate'), sample[0])
    tid = target['tweet_id']
    print(f'\\n=== Sample captions for tweet {tid} (stratum={target["stratum"]}) ===')
    print(f'  tweet_text : {target["tweet_text"][:120]}')
    print(f'  label      : {target["hard_label_binary"]} (1=hate)')
    for result_file in sorted(EVAL_DIR.glob('qwen3_8b_fp16_prompt_*.json')):
        with open(result_file, encoding='utf-8') as fh:
            data = json.load(fh)
        if tid in data and data[tid]['caption']:
            print(f'  [{result_file.stem:40s}]: {data[tid]["caption"][:180]}')
"""),

# ── 9: list output files ──────────────────────────────────────────────────────
("code", """\
print('Files in /kaggle/working/vlm_eval/:')
for f in sorted(EVAL_DIR.iterdir()):
    sz_kb = f.stat().st_size / 1024
    print(f'  {f.name:<50s}  {sz_kb:>8.1f} KB')
print('\\nDone! Download the vlm_eval/ folder from the Kaggle output tab.')
"""),

]  # end CELLS_RAW


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
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.12.12",
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

OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {OUT_NB}")
