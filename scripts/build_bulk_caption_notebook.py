"""
Generates vlm_bulk_caption_kaggle.ipynb — A4 full-scale captioning.

Correct numbers (verified 2026-05-03 after re-running A2 on full 150K):
  - ocr_consolidated_filtered.json : 150,000 entries
  - Has OCR text (skip)            : 70,744
  - No OCR text -> need VLM        : 79,256 (all have image files)
  - At ~4s/img: ~88h = ~8 Kaggle sessions of 12h

API (confirmed fast path from vram_test_corrected.ipynb):
  - AutoModelForImageTextToText, device_map='auto'
  - process_vision_info(messages, image_patch_size=...) — 2-value unpack
  - do_resize=False  (prevents 11s/img slowdown)
  - processor loaded WITHOUT min/max_pixels
  - .to(model.device)
"""
import json
from pathlib import Path

DATASET_PATH = "/kaggle/input/datasets/ekanshkhullar/updated-hate-speech-dataset/dataset"
OUT_NB = Path(__file__).parent.parent / "notebooks" / "vlm_bulk_caption_kaggle.ipynb"

CELLS_RAW = [

# ── 0: markdown ───────────────────────────────────────────────────────────────
("markdown", """\
# MMHS150K — A4 Bulk VLM Captioning (Qwen3-VL-8B FP16)

**Goal**: Generate image descriptions (Prompt 1) for all **79,256** images
that have no OCR text after A2 filtering — the only non-tweet signal for these
images in the downstream 6-class hate speech classifier.

**Input**: Your uploaded dataset (slug: `updated-hate-speech-dataset`)
containing `ocr_consolidated_filtered.json` (150K entries) and `img_resized/`

**Output**: `/kaggle/working/vlm_captions.json`
Format: `{tweet_id: {"caption": "...", "elapsed_s": 3.8}}`

**Checkpoint**: `/kaggle/working/vlm_captions_progress.json` — saved every
100 images using atomic write (temp file → rename) to prevent corruption on kill.

**Model**: Qwen3-VL-8B-Instruct FP16, `device_map='auto'` (dual T4 ~17.5GB)
**API**: fast path from `vram_test_corrected.ipynb` (~2–4s/img)

> ⚠️ At ~4s/image × 79,256 images = **~88h total = ~8 Kaggle sessions**.
> Run → checkpoint fills → download checkpoint → start next session → it resumes.
"""),

# ── 1: install ────────────────────────────────────────────────────────────────
("code", """\
!pip install -q 'transformers>=4.57.0' bitsandbytes accelerate 'qwen_vl_utils>=0.0.14' Pillow tqdm
print('done')
"""),

# ── 2: paths & config ─────────────────────────────────────────────────────────
("code", f"""\
import json, gc, os, time, tempfile
from pathlib import Path

# Expand VRAM segments to reduce fragmentation-related OOM on long runs
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# ── Dataset paths ──────────────────────────────────────────────────────────────
INPUT_DIR  = Path('{DATASET_PATH}')
WORK_DIR   = Path('/kaggle/working')

IMG_DIR    = INPUT_DIR / 'img_resized'
OCR_FILE   = WORK_DIR / 'ocr_filtered.json'                 # full 150K file from eval notebook A2

OUT_FILE   = WORK_DIR / 'vlm_captions.json'                # final output (written at end)
CKPT_FILE  = WORK_DIR / 'vlm_captions_progress.json'       # live checkpoint (every 100 imgs)
ERR_FILE   = WORK_DIR / 'vlm_captions_errors.json'         # failed IDs across sessions

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID   = 'Qwen/Qwen3-VL-8B-Instruct'

# ── Prompt 1: pure visual description, no hate commentary ────────────────────
# Concise instructions — avoids verbose output that wastes tokens and inference time.
# Critically does NOT ask about hate/offensive content, which causes the model
# to add "no hate found" boilerplate that corrupts downstream classifier training.
PROMPT = (
    'Describe what you see in this image in 1-2 short sentences. '
    'Cover the main subjects, objects, any visible text, and the setting. '
    'Be factual and concise — do not over-explain or add commentary.'
)

# ── Inference settings ─────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 100    # ~400 chars cap — keeps captions concise, inference faster
SAVE_EVERY     = 100    # checkpoint every 100 images (atomic write, crash-safe)

# ── Verify all required paths exist before proceeding ─────────────────────────
print('Path verification:')
all_ok = True
for p in [INPUT_DIR, IMG_DIR, OCR_FILE]:
    ok = p.exists()
    if not ok:
        all_ok = False
    print(f'  [{{"OK" if ok else "MISSING"}}] {{p}}')
if not all_ok:
    raise FileNotFoundError('One or more required paths are MISSING. Fix paths before continuing.')
print()
print(f'Output checkpoint : {{CKPT_FILE}}')
print(f'Final output      : {{OUT_FILE}}')
"""),

# ── 3: identify no-OCR images ─────────────────────────────────────────────────
("code", """\
print('Loading ocr_consolidated_filtered.json ...')
with open(OCR_FILE, encoding='utf-8') as f:
    ocr_filtered = json.load(f)

total_entries = len(ocr_filtered)
have_text = [tid for tid, txt in ocr_filtered.items() if str(txt).strip()]
need_text = [tid for tid, txt in ocr_filtered.items() if not str(txt).strip()]

print(f'  Total entries in OCR file  : {total_entries:,}')
print(f'  Have OCR text (skip VLM)   : {len(have_text):,}  ({len(have_text)/total_entries*100:.1f}%)')
print(f'  No OCR text -> need VLM    : {len(need_text):,}  ({len(need_text)/total_entries*100:.1f}%)')
print()

# Scan img_resized/ directory ONCE and build a set of available IDs.
# Much faster than calling path.exists() ~79K times individually
# (each .exists() is a syscall; 79K calls takes several minutes on Kaggle).
print('Scanning img_resized/ for available images (one-time scan)...')
existing_ids = {p.stem for p in IMG_DIR.iterdir() if p.suffix == '.jpg'}
print(f'  Images on disk             : {len(existing_ids):,}')

need_with_img = [tid for tid in need_text if tid in existing_ids]
missing_img   = [tid for tid in need_text if tid not in existing_ids]
print(f'  Need VLM + image exists    : {len(need_with_img):,}')
if missing_img:
    print(f'  WARNING: {len(missing_img):,} no-OCR IDs have no image on disk — will be skipped')

# Sorted for deterministic, resumable order across sessions
import math
need_ids = sorted(need_with_img)
print()
print(f'Final target : {len(need_ids):,} images to caption')
est_h    = len(need_ids) * 4.0 / 3600
sessions = math.ceil(est_h / 12)
print(f'ETA at ~4s/img: {est_h:.0f}h total = ~{sessions} Kaggle sessions of 12h')
"""),

# ── 4: resume from checkpoint ─────────────────────────────────────────────────
("code", """\
# Load checkpoint (captions saved from previous sessions)
if CKPT_FILE.exists():
    with open(CKPT_FILE, encoding='utf-8') as f:
        done_captions = json.load(f)
    print(f'[RESUME] Checkpoint: {len(done_captions):,} captions already saved')
else:
    done_captions = {}
    print('[START] No checkpoint found — starting from scratch')

# Load errors log (IDs that failed in previous sessions) — these are retried
if ERR_FILE.exists():
    with open(ERR_FILE, encoding='utf-8') as f:
        all_errors = json.load(f)
    print(f'[INFO] {len(all_errors):,} errors from previous sessions (will retry)')
else:
    all_errors = {}

# Skip IDs that are already done AND have a non-empty caption
# (empty caption = error from previous session — will be retried this session)
done_nonempty = {tid for tid, v in done_captions.items() if v['caption'].strip()}
remaining_ids = [tid for tid in need_ids if tid not in done_nonempty]

already_done = len(done_nonempty)
print()
print(f'Already captioned (non-empty) : {already_done:,} / {len(need_ids):,} '
      f'({already_done/len(need_ids)*100:.1f}%)')
print(f'Remaining this session        : {len(remaining_ids):,}')
est_remaining_h = len(remaining_ids) * 4.0 / 3600
print(f'ETA at 4s/img (12h session max): {min(est_remaining_h, 12):.1f}h')
approx_this_run = min(len(remaining_ids), int(12 * 3600 / 4))
print(f'Will caption approx           : {approx_this_run:,} images this run')
"""),

# ── 5: load model ─────────────────────────────────────────────────────────────
("code", """\
import torch
from PIL import Image
from tqdm.notebook import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ── Clear any stale GPU memory before loading ─────────────────────────────────
# Critical: re-running this cell on a dirty session causes OOM without this
for _name in ('model', 'processor'):
    try:
        exec(f'del {_name}')
    except NameError:
        pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

print(f'Loading {MODEL_ID} FP16 (device_map=auto)...')
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map='auto',
).eval()

# Load processor WITHOUT min/max_pixels — required for the do_resize=False fast path
# Setting min/max_pixels conflicts with process_vision_info pre-resizing
processor = AutoProcessor.from_pretrained(MODEL_ID)

print('GPU VRAM after model load:')
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1e9
    total_v = torch.cuda.get_device_properties(i).total_memory / 1e9
    print(f'  GPU{i}: {alloc:.1f} / {total_v:.1f} GB  ({alloc/total_v*100:.0f}% used)')
print(f'Primary inference device: {model.device}')
"""),

# ── 6: inference loop with atomic checkpoint ───────────────────────────────────
("code", """\
def caption_image(img_path: str) -> tuple[str, float]:
    \"\"\"
    Caption one image using Qwen3-VL-8B fast inference path.
    Returns (caption_str, elapsed_seconds).
    Fast path: process_vision_info with image_patch_size + do_resize=False
    Confirmed ~2-4s/img vs ~11s with naive resizing.
    \"\"\"
    img = Image.open(img_path).convert('RGB')
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': img},
        {'type': 'text',  'text':  PROMPT},
    ]}]

    text_in = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2-value unpack with image_patch_size — confirmed fast path
    # process_vision_info pre-resizes to patch grid, then do_resize=False
    # skips the redundant processor resize that caused 11s/img slowdown
    image_inputs, video_inputs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
    )
    inputs = processor(
        text=[text_in],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        do_resize=False,
        return_tensors='pt',
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0

    prompt_len = inputs['input_ids'].shape[1]
    caption = processor.decode(
        out_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()
    # Immediately free GPU tensors — don't wait for Python GC
    del inputs, out_ids
    return caption, round(elapsed, 3)


def atomic_save(data: dict, path: Path):
    \"\"\"
    Write JSON atomically via temp file + rename.
    Prevents checkpoint corruption if session is killed mid-write.
    \"\"\"
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.rename(path)   # atomic on Linux (Kaggle runs Linux)


# ── Main captioning loop ───────────────────────────────────────────────────────
session_errors  = {}
session_times   = []
session_count   = 0

print(f'Starting: {len(remaining_ids):,} images to caption this session')
print(f'Checkpoint every {SAVE_EVERY} images -> {CKPT_FILE.name}  (atomic write)')
print('-' * 65)

for idx, tid in enumerate(tqdm(remaining_ids, desc='A4 VLM captioning')):
    img_path = IMG_DIR / f'{tid}.jpg'
    try:
        caption, elapsed = caption_image(str(img_path))
        done_captions[tid] = {'caption': caption, 'elapsed_s': elapsed}
        session_times.append(elapsed)
        session_count += 1
    except Exception as e:
        err_msg = str(e)
        session_errors[tid] = err_msg
        # Save empty placeholder so it's visible in checkpoint
        # but ISN'T counted as done_nonempty — will be retried next session
        done_captions[tid] = {'caption': '', 'elapsed_s': 0.0}

    # ── Atomic checkpoint every SAVE_EVERY images ─────────────────────────────
    if (idx + 1) % SAVE_EVERY == 0:
        atomic_save(done_captions, CKPT_FILE)

        # Update errors log (persists across sessions)
        all_errors.update(session_errors)
        if all_errors:
            atomic_save(all_errors, ERR_FILE)

        # ── Periodic VRAM cleanup ──────────────────────────────────────────────
        # Prevents CUDA memory fragmentation hang that occurs after 200-400
        # iterations. Runs every SAVE_EVERY images — no speed impact since
        # we're already pausing for the checkpoint file write.
        gc.collect()
        torch.cuda.empty_cache()

        # Progress report
        if session_times:
            recent_avg = sum(session_times[-SAVE_EVERY:]) / min(len(session_times), SAVE_EVERY)
            remaining_this = len(remaining_ids) - (idx + 1)
            eta_h = remaining_this * recent_avg / 3600
            total_done_now = len({k for k, v in done_captions.items() if v['caption'].strip()})
            pct = total_done_now / len(need_ids) * 100
            alloc0 = torch.cuda.memory_allocated(0) / 1e9
            alloc1 = torch.cuda.memory_allocated(1) / 1e9 if torch.cuda.device_count() > 1 else 0
            print(f'  [{idx+1:>6}/{len(remaining_ids):,}] ckpt saved | '
                  f'avg={recent_avg:.2f}s | ETA~{eta_h:.1f}h | '
                  f'overall={total_done_now:,}/{len(need_ids):,} ({pct:.1f}%) | '
                  f'VRAM={alloc0:.1f}+{alloc1:.1f}GB | errors={len(session_errors)}')


# ── Final save at end of session ──────────────────────────────────────────────
atomic_save(done_captions, CKPT_FILE)
all_errors.update(session_errors)
if all_errors:
    atomic_save(all_errors, ERR_FILE)

# ── Session summary ───────────────────────────────────────────────────────────
total_nonempty = len({k for k, v in done_captions.items() if v['caption'].strip()})
still_left     = len(need_ids) - total_nonempty

print()
print('=' * 65)
print('SESSION COMPLETE')
print('=' * 65)
print(f'  Captioned this session : {session_count:,}')
print(f'  Errors this session    : {len(session_errors):,}')
if session_times:
    print(f'  Avg inference time     : {sum(session_times)/len(session_times):.2f}s/image')
print()
print(f'  Overall progress : {total_nonempty:,} / {len(need_ids):,} ({total_nonempty/len(need_ids)*100:.1f}%)')
print(f'  Still remaining  : {still_left:,}')
if still_left > 0:
    print()
    print('  -> Upload checkpoint to Kaggle input dataset and start next session')
    print('     OR just restart this notebook — CKPT_FILE persists in /kaggle/working/')
else:
    print()
    print('  -> ALL DONE! Run the finalize cell below to write vlm_captions.json')
"""),

# ── 7: finalize output ────────────────────────────────────────────────────────
("code", """\
# ── Finalize: write vlm_captions.json when all sessions are complete ──────────
# This cell is safe to run at any point — it reports progress if not done yet.

# Reload checkpoint in case this is a fresh session after the last batch
if not CKPT_FILE.exists():
    print('[ERROR] No checkpoint file found. Run cells 1-6 first.')
else:
    with open(CKPT_FILE, encoding='utf-8') as f:
        done_captions = json.load(f)

    # Recompute need_ids independently (no dependency on earlier cells)
    print('Recomputing need_ids from OCR file...')
    with open(OCR_FILE, encoding='utf-8') as f:
        _ocr = json.load(f)
    need_ids = sorted(tid for tid, txt in _ocr.items()
                      if not str(txt).strip() and (IMG_DIR / f'{tid}.jpg').exists())

    total_need    = len(need_ids)
    done_nonempty = {tid for tid, v in done_captions.items() if v['caption'].strip()}
    total_done    = len(done_nonempty)
    still_left    = total_need - total_done
    empty_cap     = sum(1 for v in done_captions.values() if not v['caption'].strip())

    print(f'  Need captions : {total_need:,}')
    print(f'  Done (filled) : {total_done:,}  ({total_done/total_need*100:.1f}%)')
    print(f'  Empty/errors  : {empty_cap:,}')
    print(f'  Still left    : {still_left:,}')
    print()

    if still_left > 0:
        print(f'[NOT DONE] {still_left:,} images still need captioning.')
        print('Start another Kaggle session — the notebook will resume from checkpoint.')
    else:
        # Write final output
        atomic_save(done_captions, OUT_FILE)
        all_t = [v['elapsed_s'] for v in done_captions.values() if v['elapsed_s'] > 0]

        print(f'vlm_captions.json written -> {OUT_FILE}')
        print(f'  Total    : {len(done_captions):,}')
        print(f'  Filled   : {total_done:,}  ({total_done/len(done_captions)*100:.1f}%)')
        print(f'  Empty    : {empty_cap:,}')
        if all_t:
            print(f'  Avg time : {sum(all_t)/len(all_t):.2f}s/image')
        print()
        print('Files in /kaggle/working/:')
        for fp in sorted(WORK_DIR.iterdir()):
            sz_mb = fp.stat().st_size / 1e6
            print(f'  {fp.name:<50s}  {sz_mb:>7.1f} MB')
        print()
        print('Download vlm_captions.json from the Kaggle output tab.')
        print('Then run: python scripts/merge_vlm_captions.py')
"""),

# ── 8: merge instructions ─────────────────────────────────────────────────────
("markdown", """\
## After all sessions — merge locally

```bash
python scripts/merge_vlm_captions.py \\
    --ocr   dataset/ocr_consolidated_filtered.json \\
    --vlm   vlm_captions.json \\
    --out   dataset/unified_text.json
```

### Coverage after merge (verified numbers)

| Source | Images | % |
|---|---|---|
| OCR text (filtered, A2) | 70,744 | 47.2% |
| VLM caption P1 (A4) | 79,256 | 52.8% |
| **Total** | **150,000** | **100%** |

### Multi-session workflow

| Session | Action |
|---|---|
| 1 | Run all cells → runs ~10,800 images (12h @ 4s) → checkpoint saved |
| 2–7 | Re-run notebook → cell 4 resumes from checkpoint automatically |
| 8 | `len(done_captions) == 79,256` → run finalize cell → download |

### What's next (Phase D1)
Train RoBERTa-based multimodal classifier:
- `tweet_text` → primary hate signal
- `ocr_text` → text embedded in image
- `vlm_caption` → visual context (only for the 52.8% with no OCR)
- 6-class labels: `NotHate / Racist / Sexist / Homophobe / Religion / OtherHate`
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
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
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
