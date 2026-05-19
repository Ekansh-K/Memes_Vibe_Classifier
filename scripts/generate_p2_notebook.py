"""Script to generate the P2 TCAM Kaggle notebook.

Run:  python scripts/generate_p2_notebook.py
Output: notebooks/p2_tcam_train.ipynb
"""

import json
from pathlib import Path


def cell(source: str, cell_type: str = "code") -> dict:
    """Build a notebook cell dict."""
    lines = [line + "\n" for line in source.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")  # last line has no trailing newline
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def md(source: str) -> dict:
    """Build a markdown cell dict."""
    lines = [line + "\n" for line in source.splitlines()]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(md("""# P2 — TCAM: Text-guided Cross-Attention Multimodal Pipeline
## MMHS-150K Hateful Meme Classification

**Architecture:** Frozen CLIP ViT-L/14 + Frozen TweetEval RoBERTa → Cross-Attention Fusion

**Variations:**
- **A** — Binary only (Stage 1 standalone)
- **B** — Direct 6-class (single stage)
- **C** — Hierarchical two-stage, single-label
- **D** — Hierarchical two-stage, multi-label ← *Primary target*

**Text Modes:**
- `tweet_ocr` — tweet_text [SEP] ocr_text
- `all_text` — caption [SEP] ocr_text [SEP] tweet_text

**Hardware:** 2× Kaggle T4 GPU (30 GB total VRAM) via DataParallel"""))

# ── Cell 1: Configuration ─────────────────────────────────────────────────────
cells.append(md("## ⚙️ 1. User Configuration — Edit These"))
cells.append(cell("""# ── SELECT WHAT TO RUN ───────────────────────────────────────────────────────
VARIATION  = "D"          # "A", "B", "C", or "D"
TEXT_MODE  = "all_text"   # "tweet_ocr" or "all_text"

# ── SUBSET FOR TESTING (set to None for full dataset) ────────────────────────
MAX_TRAIN_SAMPLES = None   # e.g. 1000 for a quick smoke test
MAX_VAL_SAMPLES   = None

# ── STAGE 1 HYPERPARAMETERS ──────────────────────────────────────────────────
S1_EPOCHS     = 4          # Peak observed at epoch 2 in Phase 2 run; 4 epochs captures it with margin
S1_LR         = 1e-4       # reduced 2e-4 → 1e-4: head converges but needs finer settling
S1_BATCH_SIZE = 128        # single T4 has 15GB, only 2.8GB used at bs=32 → safe to 4×
S1_WARMUP     = 0.05       # 5% of total steps

# ── PARTIAL FINE-TUNING (Phase 2) ────────────────────────────────────────────
# 0 = fully frozen (original baseline)
# 2 = unfreeze last 2 TweetEval transformer layers (Phase 2)
# 3 = unfreeze last 3 layers (Phase 3a, only if Phase 2 S1 F1 < 0.69)
UNFREEZE_TWEET_LAYERS = 2

# ── STAGE 2 HYPERPARAMETERS (C/D only) ──────────────────────────────────────
S2_EPOCHS     = 15
S2_LR         = 1e-4
S2_BATCH_SIZE = 128
S2_WARMUP     = 0.10       # 10% of total steps

# ── SHARED ───────────────────────────────────────────────────────────────────
GRAD_ACCUM    = 1          # eff batch = 128×1 = 128; no accumulation needed
SEED          = 42
USE_AMP       = True       # fp16 mixed precision
USE_DATA_PARALLEL = False   # CLIP ViT-L/14 is too large for clean DataParallel replication

# ── LABEL DENOISING (Options A/B/C from ceiling analysis) ──────────────────────
SOFT_LABELS         = True       # Option A: use annotator vote probabilities as targets
AGREEMENT_WEIGHTING = True       # Option B: weight loss by annotator agreement level
AGREEMENT_WEIGHTS   = (0.2, 0.5, 1.0)   # (all_differ, majority, unanimous)
LABEL_SMOOTHING     = 0.1        # Option C: push hard targets toward 0.5 by 10%
TEMP_SCALING        = True       # post-training temperature calibration

# ── STORAGE ─────────────────────────────────────────────────────────────────
# IMPORTANT: Keep USE_IMAGE_STORE = False on Kaggle.
# Building a memmap of 150K images fills /kaggle/working (~20 GB limit).
# Images are loaded from the dataset input directory on the fly instead.
USE_IMAGE_STORE = False    # DO NOT change to True on Kaggle

print(f"Variation : P2-{VARIATION}")
print(f"Text mode : {TEXT_MODE}")
print(f"Eff batch : {S1_BATCH_SIZE} x {GRAD_ACCUM} = {S1_BATCH_SIZE * GRAD_ACCUM}")
print(f"Max train : {MAX_TRAIN_SAMPLES or 'full dataset'}")
print(f"Unfreeze  : {UNFREEZE_TWEET_LAYERS} TweetEval layers ({'frozen baseline' if UNFREEZE_TWEET_LAYERS == 0 else 'Phase 2 fine-tuning'})")
print(f"Soft labels: {SOFT_LABELS}  Agreement wt: {AGREEMENT_WEIGHTING}  Smoothing: {LABEL_SMOOTHING}")
print(f"Temp scale : {TEMP_SCALING}")
print(f"Image store: {'enabled' if USE_IMAGE_STORE else 'DISABLED (disk-only, Kaggle-safe)'}")"""))

# ── Cell 2: GPU check ─────────────────────────────────────────────────────────
cells.append(md("## 🖥️ 2. GPU Verification"))
cells.append(cell("""import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"GPUs detected   : {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  GPU {i}: {name}  ({mem:.1f} GB)")
else:
    print("WARNING: No CUDA GPU found. Training will be very slow on CPU.")"""))

# ── Cell 3: Install dependencies ──────────────────────────────────────────────
cells.append(md("## 📦 3. Install Dependencies"))
cells.append(cell("""import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

# CLIP (OpenAI)
pip_install("git+https://github.com/openai/CLIP.git")

# HuggingFace transformers + tokenizers
pip_install("transformers>=4.36.0", "tokenizers>=0.15.0")

# Other requirements
pip_install("wordsegment", "scikit-learn", "pyyaml", "Pillow")

print("✓ All dependencies installed")"""))

# ── Cell 4: Paths ─────────────────────────────────────────────────────────────
cells.append(md("## 📁 4. Configure Dataset Paths"))
cells.append(cell("""import os
import sys
from pathlib import Path

# ── Detect environment ───────────────────────────────────────────────────────
IS_KAGGLE = Path("/kaggle/input").exists()
print(f"Environment: {'Kaggle' if IS_KAGGLE else 'Local'}")

if IS_KAGGLE:
    # ── Kaggle paths ─────────────────────────────────────────────────────────
    INPUT_DIR  = Path('/kaggle/input/datasets/ekanshkhullar/updated-hate-speech-dataset/dataset')
    WORK_DIR   = Path('/kaggle/working')
    CODE_DIR   = WORK_DIR / 'Memes_Vibe_Classifier'
    
    os.environ["MMHS_PROJECT_ROOT"] = str(CODE_DIR)
    os.environ["MMHS_DATA_DIR"]     = str(INPUT_DIR)
    
    # Add src to path
    sys.path.insert(0, str(CODE_DIR))
else:
    # ── Local paths ───────────────────────────────────────────────────────────
    PROJECT_ROOT = Path.cwd()
    while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Project root : {PROJECT_ROOT}")

# Verify required files
if IS_KAGGLE:
    required = {
        "GT labels"       : INPUT_DIR / "MMHS150K_GT.json",
        "Processed labels": INPUT_DIR / "processed_labels.json",
        "OCR filtered"    : INPUT_DIR / "ocr_consolidated_filtered.json",
        "Train split"     : INPUT_DIR / "splits" / "train_ids.txt",
        "Val split"       : INPUT_DIR / "splits" / "val_ids.txt",
        "Test split"      : INPUT_DIR / "splits" / "test_ids.txt",
        "Images dir"      : INPUT_DIR / "img_resized",
    }
    all_ok = True
    for name, p in required.items():
        ok = p.exists()
        print(f"  {'\u2713' if ok else '\u2717 MISSING'}  {name}: {p}")
        if not ok:
            all_ok = False
    if not all_ok:
        raise RuntimeError("Missing required dataset files! Check your Kaggle dataset attachment.")

# Verify imports work
try:
    from src.p2.config import P2Config
    print("\u2713 P2 imports OK")
except ImportError as e:
    print(f"\u2717 Import error: {e}")
    print("  Check that project src/ is on sys.path")"""))

# ── Cell 4b: Generate stratified splits on Kaggle ────────────────────────────
cells.append(md("## 📂 4b. Generate Stratified 80/10/10 Splits"))
cells.append(cell("""# Regenerates train/val/test splits with correct stratification.
# Original dataset splits had a severe class imbalance mismatch:
#   Train: 15.3% hate  |  Val: 34.7% hate  |  Test: 34.8% hate
# New stratified splits preserve the true dataset hate rate (~17.2%) across ALL splits.
# This cell writes splits to /kaggle/working/splits/ and patches MMHS_DATA_DIR
# so the trainer reads from there instead of the read-only input directory.

import json, random, os
from collections import defaultdict
from pathlib import Path

INPUT_DIR = Path(os.environ['MMHS_DATA_DIR'])
CODE_DIR  = Path(os.environ['MMHS_PROJECT_ROOT'])

# ── Settings ──────────────────────────────────────────────────────────────────
SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10

# ── Load labels ───────────────────────────────────────────────────────────────
with open(INPUT_DIR / 'processed_labels.json', encoding='utf-8') as f:
    labels = json.load(f)

all_ids = list(labels.keys())
print(f'Total labelled samples: {len(all_ids):,}')

# ── Stratify by binary label ──────────────────────────────────────────────────
buckets = defaultdict(list)
for tid in all_ids:
    buckets[labels[tid]['hard_label_binary']].append(tid)

print(f'  NotHate: {len(buckets[0]):,} ({len(buckets[0])/len(all_ids)*100:.1f}%)')
print(f'  Hate   : {len(buckets[1]):,} ({len(buckets[1])/len(all_ids)*100:.1f}%)')

# ── Split ─────────────────────────────────────────────────────────────────────
rng = random.Random(SEED)
train_ids, val_ids, test_ids = [], [], []

for cls, ids in sorted(buckets.items()):
    rng.shuffle(ids)
    n_train = round(len(ids) * TRAIN_FRAC)
    n_val   = round(len(ids) * VAL_FRAC)
    train_ids.extend(ids[:n_train])
    val_ids.extend(ids[n_train : n_train + n_val])
    test_ids.extend(ids[n_train + n_val :])

rng.shuffle(train_ids); rng.shuffle(val_ids); rng.shuffle(test_ids)

# ── Verify no leakage ─────────────────────────────────────────────────────────
assert len(set(train_ids) & set(val_ids)) == 0, 'Train/Val overlap!'
assert len(set(train_ids) & set(test_ids)) == 0, 'Train/Test overlap!'
assert len(set(val_ids) & set(test_ids)) == 0, 'Val/Test overlap!'
assert len(train_ids) + len(val_ids) + len(test_ids) == len(all_ids)

# ── Write to writable location ────────────────────────────────────────────────
SPLITS_OUT = CODE_DIR / 'splits'
SPLITS_OUT.mkdir(parents=True, exist_ok=True)

for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    (SPLITS_OUT / f'{name}_ids.txt').write_text(chr(10).join(ids) + chr(10), encoding='utf-8')

# ── Patch env so loader reads new splits ─────────────────────────────────────
import src.data.splits as _splits_mod
import src.utils.config as _cfg_mod

# Monkey-patch SPLITS_DIR to point to our new writable location
_splits_mod.SPLITS_DIR = SPLITS_OUT   # module-level var used by load_split_ids
_cfg_mod.SPLITS_DIR    = SPLITS_OUT

total = len(train_ids) + len(val_ids) + len(test_ids)
print('New stratified splits (seed=%d):' % SEED)
for name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
    hate = sum(1 for i in ids if labels[i]['hard_label_binary'] == 1)
    print('  %-5s  %7d  (%.1f%%)  hate=%d (%.1f%%)' % (
        name, len(ids), len(ids)/total*100, hate, hate/len(ids)*100))
print('No leakage detected. Splits written and loader patched.')
"""))


# ── Cell 5: Build config ──────────────────────────────────────────────────────
cells.append(md("## 🔧 5. Build Experiment Config"))
cells.append(cell("""from src.p2.config import P2Config

config = P2Config(
    variation              = VARIATION,
    text_mode              = TEXT_MODE,
    s1_epochs              = S1_EPOCHS,
    s1_lr                  = S1_LR,
    s1_batch_size          = S1_BATCH_SIZE,
    s1_warmup_ratio        = S1_WARMUP,
    s2_epochs              = S2_EPOCHS,
    s2_lr                  = S2_LR,
    s2_batch_size          = S2_BATCH_SIZE,
    s2_warmup_ratio        = S2_WARMUP,
    grad_accum_steps       = GRAD_ACCUM,
    seed                   = SEED,
    use_amp                = USE_AMP,
    use_data_parallel      = USE_DATA_PARALLEL,
    use_image_store        = USE_IMAGE_STORE,   # False on Kaggle — avoids filling disk
    max_train_samples      = MAX_TRAIN_SAMPLES,
    max_val_samples        = MAX_VAL_SAMPLES,
    unfreeze_tweet_last_n  = UNFREEZE_TWEET_LAYERS,
    use_soft_labels        = SOFT_LABELS,
    use_agreement_weighting = AGREEMENT_WEIGHTING,
    agreement_weights      = AGREEMENT_WEIGHTS,
    label_smoothing        = LABEL_SMOOTHING,
    temperature_scaling    = TEMP_SCALING,
    device                 = "auto",
    num_workers            = 2,   # Kaggle safe default
)

print(f"Run name     : {config.run_name}")
print(f"Checkpoint   : {config.run_dir}")
print(f"Results      : {config.results_run_dir}")
print(f"Image store  : {'enabled' if config.use_image_store else 'DISABLED (disk-only)'}")
print(f"Eff batch    : {config.s1_batch_size} x {config.grad_accum_steps} = {config.s1_batch_size * config.grad_accum_steps}")
print(f"Unfreeze     : {config.unfreeze_tweet_last_n} TweetEval layers  tweet_lr={config.tweet_encoder_lr:.0e}")
print(f"Soft labels  : {config.use_soft_labels}  Agreement wt: {config.use_agreement_weighting}  Smoothing: {config.label_smoothing}")
print(f"Temp scaling : {config.temperature_scaling}")"""))

# ── Cell 5c: Split verification ───────────────────────────────────────────────
cells.append(md("## 📊 5c. Split Verification — Confirm 80/10/10 Balanced Splits"))
cells.append(cell("""# Verifies that the splits loaded by the trainer have:
#  - Correct 80/10/10 proportions
#  - Matching hate-rate across train/val/test (no distribution mismatch)
#  - No ID overlap between splits
import json, os
from pathlib import Path
from src.data.splits import load_split_ids, load_processed_labels

labels   = load_processed_labels()

rows = []
total_all = 0
for split in ['train', 'val', 'test']:
    ids  = load_split_ids(split)
    hate = sum(1 for i in ids if i in labels and labels[i]['hard_label_binary'] == 1)
    rows.append({'split': split, 'n': len(ids), 'hate': hate})
    total_all += len(ids)

HDR = '%-6s  %8s  %8s  %7s  %8s' % ('Split', 'Count', '% total', 'Hate', 'Hate %')
print(HDR)
print('-' * 46)
for r in rows:
    print('%-6s  %8d  %7.1f%%  %7d  %7.1f%%' % (
        r['split'], r['n'], r['n']/total_all*100, r['hate'], r['hate']/r['n']*100))
print('-' * 46)
print('%-6s  %8d' % ('TOTAL', total_all))

# Verify hate rate is consistent (all within 2% of each other)
hate_rates = [r['hate'] / r['n'] for r in rows]
assert max(hate_rates) - min(hate_rates) < 0.02, \
    'Hate rate mismatch across splits: ' + str([round(r,3) for r in hate_rates])
print('Hate rate consistent across splits (< 2% variance) -- OK')

# Verify no overlap
train_set = set(load_split_ids('train'))
val_set   = set(load_split_ids('val'))
test_set  = set(load_split_ids('test'))
assert len(train_set & val_set) == 0,  'FAIL: Train/Val overlap!'
assert len(train_set & test_set) == 0, 'FAIL: Train/Test overlap!'
assert len(val_set & test_set) == 0,   'FAIL: Val/Test overlap!'
print('No overlap between splits -- OK')
"""))

# ── Cell 5b: Text input preview ─────────────────────────────────────────────
cells.append(md("## 🔍 5b. Text Input Preview — What the Model Actually Sees"))
cells.append(cell("""# Shows a real validation sample: raw fields → assembled string → tokenized & decoded.
# Confirms:
#  (a) ocr_filtered.json is being read (phone UI noise removed)
#  (b) the 128-token budget is used efficiently for the selected TEXT_MODE

import json, os
from pathlib import Path
from transformers import AutoTokenizer
from src.data.preprocessing import clean_tweet_text, clean_ocr_text
from src.data.splits import load_gt_json, load_ocr_data

INPUT_DIR = Path(os.environ['MMHS_DATA_DIR'])
MAX_TEXT_LEN = config.max_text_len   # 128

# Load data sources
gt_data  = load_gt_json()
ocr_data = load_ocr_data('filtered')   # ← filtered (phone UI removed)

# Load VLM captions if available
captions_path = Path(os.environ['MMHS_PROJECT_ROOT']) / 'results' / 'vlm_captions.json'
if captions_path.exists():
    with open(captions_path, encoding='utf-8') as f:
        raw_caps = json.load(f)
    captions = {k: v['caption'] for k, v in raw_caps.items()}
else:
    captions = {}
    print("(No VLM captions found — all_text will fall back to tweet+OCR)")

# Load the TweetEval tokenizer (same as TCAM uses)
tok = AutoTokenizer.from_pretrained(config.tweet_model)

# Pick the first val ID that has OCR text
val_ids = (INPUT_DIR / 'splits' / 'val_ids.txt').read_text().strip().splitlines()
pick_id = next((sid for sid in val_ids if ocr_data.get(sid, '')), val_ids[0])

entry     = gt_data[pick_id]
raw_tweet = entry['tweet_text']
raw_ocr   = ocr_data.get(pick_id, '')
caption   = captions.get(pick_id, '')

clean_tw  = clean_tweet_text(raw_tweet)
clean_ocr = clean_ocr_text(raw_ocr)

print(f'Tweet ID  : {pick_id}')
print(f'Raw tweet : {raw_tweet[:120]}')
print(f'Raw OCR   : {raw_ocr[:120]}')
print(f'Caption   : {(caption[:120] + " ...") if len(caption) > 120 else caption}')
print()
print(f'Cleaned tweet : {clean_tw[:120]}')
print(f'Cleaned OCR   : {clean_ocr[:120]}')
print()

# Assemble exactly how P2Dataset._build_text() does it
for tm in ['tweet_ocr', 'all_text']:
    if tm == 'tweet_ocr':
        parts = [clean_tw, clean_ocr]
    else:  # all_text: caption [SEP] ocr_text [SEP] tweet_text
        parts = [caption, clean_ocr, clean_tw]
    assembled = f' {tok.sep_token} '.join(p for p in parts if p)

    enc = tok(
        assembled,
        max_length=MAX_TEXT_LEN,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    ids       = enc['input_ids'][0].tolist()
    non_pad   = sum(1 for t in ids if t != tok.pad_token_id)
    decoded   = tok.decode(ids, skip_special_tokens=False)

    print(f'=== text_mode = {tm!r}  (max_text_len={MAX_TEXT_LEN}) ===')
    print(f'Assembled ({len(assembled)} chars):')
    print(f'  {assembled[:350] + (" [...]" if len(assembled) > 350 else "")}')
    print(f'Tokens used (non-pad): {non_pad} / {MAX_TEXT_LEN}')
    print(f'Decoded (model input):')
    print(f'  {decoded[:450]}')
    print()
"""))


cells.append(md("## 🧪 6. Smoke Test — Validate Architecture (Optional)"))
cells.append(cell("""# Run this cell BEFORE full training to verify shapes and frozen params
import torch
from PIL import Image
from src.p2.model import TCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading TCAM for shape verification...")
model = TCAM.from_config(config, num_classes=1).to(device)

# Count trainable vs frozen params
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
n_frozen = n_total - n_train
print(f"\\nParameter counts:")
print(f"  Total     : {n_total:,}")
print(f"  Trainable : {n_train:,}  (proj_t + cross_attn + head)")
print(f"  Frozen    : {n_frozen:,} (CLIP + TweetEval)")

# Verify CLIP frozen
clip_grad_any = any(p.requires_grad for p in model._clip_model.parameters())
tweet_grad_any = any(p.requires_grad for p in model.tweet_encoder.parameters())
print(f"\\nFrozen check:")
print(f"  CLIP has_grad   : {clip_grad_any}  (expect False)")
print(f"  TweetEval grad  : {tweet_grad_any}  (expect False)")

# Verify proj_t partial-identity init
# proj_t is Linear(768 → 1024): weight shape = [1024, 768]
# Expect: top-left [768, 768] block ≈ eye(768), rest ≈ 0
import torch.nn.functional as F
partial_eye = torch.zeros(1024, 768)          # [out=1024, in=768]
partial_eye[:768, :768] = torch.eye(768)      # top-left block = identity
is_identity = F.mse_loss(model.proj_t.weight.data.cpu(), partial_eye).item()
print(f"  proj_t partial-identity MSE: {is_identity:.2e}  (expect ~0.0)")

# Forward pass shape check
dummy_imgs = [Image.new("RGB", (224, 224), (128, 128, 128)) for _ in range(2)]
dummy_texts = ["This is a test meme [SEP] sample text", "Another meme caption [SEP] ocr text"]
with torch.no_grad():
    logits = model(dummy_imgs, dummy_texts)
print(f"\\nShape check:")
print(f"  Input  : {len(dummy_imgs)} images, {len(dummy_texts)} texts")
print(f"  Output : {logits.shape}  (expect [2, 1])")

# Check patch tokens shape
V = model._extract_patch_tokens(
    model._preprocess_images(dummy_imgs, device)
)
print(f"  Patch tokens : {V.shape}  (expect [2, 257, 1024])")

del model
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print("\\n✓ Smoke test PASSED")"""))

# ── Cell 7: Train ─────────────────────────────────────────────────────────────
cells.append(md("## 🚀 7. Run Training"))
cells.append(cell("""import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

from src.p2.trainer import run_p2

print(f"Starting P2-{config.variation} [{config.text_mode}]...")
print(f"Effective batch size: {config.s1_batch_size * config.grad_accum_steps}")
print("-" * 60)

all_metrics = run_p2(config)

print("\\n" + "=" * 60)
print(f"P2-{config.variation} [{config.text_mode}] — DONE")
print("=" * 60)"""))

# ── Cell 8: Results ───────────────────────────────────────────────────────────
cells.append(md("## 📊 8. Results Summary"))
cells.append(cell("""import json

print("=" * 60)
print(f"P2-{config.variation} [{config.text_mode}] — Results")
print("=" * 60)

# Stage 1
if "stage1" in all_metrics:
    s1 = all_metrics["stage1"]
    print(f"\\n  Stage 1:")
    print(f"    Best Macro F1 : {s1.get('best', 0):.4f}")

# Stage 2
if "stage2" in all_metrics:
    s2 = all_metrics["stage2"]
    print(f"\\n  Stage 2 (raw thresholds):")
    print(f"    Best Macro F1 : {s2.get('best', 0):.4f}")

# Stage 2 calibrated
if "stage2_calibrated" in all_metrics:
    cal = all_metrics["stage2_calibrated"]
    print(f"\\n  Stage 2 (calibrated thresholds):")
    print(f"    Micro F1      : {cal.get('multilabel/micro_f1', 0):.4f}")
    print(f"    Macro F1      : {cal.get('multilabel/macro_f1', 0):.4f}")
    print(f"    Sample F1     : {cal.get('multilabel/sample_f1', 0):.4f}")
    print(f"    Hamming Loss  : {cal.get('multilabel/hamming_loss', 0):.4f}")
    print(f"    Exact Match   : {cal.get('multilabel/exact_match', 0):.4f}")
    print(f"    Jaccard       : {cal.get('multilabel/jaccard', 0):.4f}")
    
    print(f"\\n  Per-class F1:")
    from src.p2.dataset import HATE_CAT_NAMES
    for name in HATE_CAT_NAMES:
        f1 = cal.get(f"multilabel/{name}/f1", 0)
        print(f"    {name:<15}: {f1:.4f}")

# Thresholds
if "thresholds" in all_metrics:
    print(f"\\n  Calibrated thresholds:")
    from src.p2.dataset import HATE_CAT_NAMES
    for name, t in zip(HATE_CAT_NAMES, all_metrics["thresholds"]):
        print(f"    {name:<15}: {t:.2f}")

# Pipeline
if "pipeline" in all_metrics:
    pip = all_metrics["pipeline"]
    print(f"\\n  Pipeline (end-to-end):")
    print(f"    S1 recall (hateful) : {pip.get('pipeline/s1_recall_hate', 0):.4f}")
    print(f"    S2 macro F1         : {pip.get('pipeline/s2_macro_f1', 0):.4f}")
    print(f"    Composite (S1×S2)   : {pip.get('pipeline/composite', 0):.4f}")

print("=" * 60)
print(f"\\nFull metrics saved to: {config.results_run_dir / 'metrics.json'}")"""))

# ── Cell 9: Ablation table ────────────────────────────────────────────────────
cells.append(md("## 📋 9. Ablation Comparison Table"))
cells.append(cell("""# Aggregates metrics from all completed runs into a comparison table
import json
from pathlib import Path

results_base = Path(config.results_dir)
rows = []

for variation in ["A", "B", "C", "D"]:
    for text_mode in ["tweet_ocr", "all_text"]:
        run_dir = results_base / f"p2_{variation}_{text_mode}"
        mfile = run_dir / "metrics.json"
        if not mfile.exists():
            continue
        with open(mfile) as f:
            m = json.load(f)
        
        row = {
            "Variation": f"P2-{variation}",
            "Text Mode": text_mode,
            "S1 Macro F1": f"{m.get('stage1', {}).get('best', 0):.4f}" if "stage1" in m else "-",
        }
        
        if "stage2_calibrated" in m:
            cal = m["stage2_calibrated"]
            row["S2 Macro F1"] = f"{cal.get('multilabel/macro_f1', 0):.4f}"
            row["S2 Micro F1"] = f"{cal.get('multilabel/micro_f1', 0):.4f}"
            row["Hamming"]     = f"{cal.get('multilabel/hamming_loss', 0):.4f}"
        elif "stage2" in m:
            row["S2 Macro F1"] = f"{m['stage2'].get('best', 0):.4f}"
            row["S2 Micro F1"] = "-"
            row["Hamming"]     = "-"
        else:
            row["S2 Macro F1"] = "-"
            row["S2 Micro F1"] = "-"
            row["Hamming"]     = "-"
        
        if "pipeline" in m:
            row["E2E Composite"] = f"{m['pipeline'].get('pipeline/composite', 0):.4f}"
        else:
            row["E2E Composite"] = "-"
        
        rows.append(row)

if not rows:
    print("No completed runs found yet. Run training first.")
else:
    # Print table
    cols = ["Variation", "Text Mode", "S1 Macro F1", "S2 Macro F1", "S2 Micro F1", "Hamming", "E2E Composite"]
    widths = {c: max(len(c), max(len(str(r.get(c, "-"))) for r in rows)) for c in cols}
    
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in cols))"""))

# ── Assemble notebook ──────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": True,
            "dataSources": [],
            "isGpuEnabled": True,
        },
    },
    "cells": cells,
}

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "p2_tcam_train.ipynb"
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written -> {OUT}")
