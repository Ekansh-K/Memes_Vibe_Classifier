#!/usr/bin/env python3
"""
Build the Kaggle P7 notebook: p7_mhsdf_kaggle.ipynb

Run this script ONCE locally to generate the notebook:
    python scripts/build_kaggle_notebook.py

Then upload the resulting notebooks/p7_mhsdf_kaggle.ipynb to Kaggle.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "notebooks" / "p7_mhsdf_kaggle.ipynb"

# ── Cell helpers ──────────────────────────────────────────────────────────────

def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip(),
    }

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip(),
    }

# ── Cell sources ──────────────────────────────────────────────────────────────

CELL_0_INSTALL = """
# Cell 0: Install dependencies + download GloVe Twitter 200d
# ── Run this cell first. Internet must be ON in Kaggle settings. ──────────────

import subprocess, os
subprocess.run([
    "pip", "install", "-q",
    "transformers>=4.41.0",
    "tokenizers",
    "scikit-learn",
    "pyyaml",
    "Pillow",
    "wordsegment",
    "emoji"
], check=True)
print("Packages installed.")

# GloVe Twitter 200d  (~1.5GB download, ~5min)
GLOVE_PATH = "/kaggle/working/glove.twitter.27B.200d.txt"
if not os.path.exists(GLOVE_PATH):
    print("Downloading GloVe Twitter 200d ...")
    os.system("wget -q https://nlp.stanford.edu/data/glove.twitter.27B.zip -O /kaggle/working/glove.zip")
    os.system("unzip -q /kaggle/working/glove.zip glove.twitter.27B.200d.txt -d /kaggle/working/")
    os.remove("/kaggle/working/glove.zip")
    print(f"GloVe ready at {GLOVE_PATH}")
else:
    print(f"GloVe already present: {GLOVE_PATH}")
"""

CELL_1_ENV = """
# Cell 1: Environment + path setup
# MUST run before any src.* import so env vars are visible to utils/config.py

import sys, os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
INPUT_DIR = Path('/kaggle/input/datasets/ekanshkhullar/updated-hate-speech-dataset/dataset')
WORK_DIR  = Path('/kaggle/working')
CODE_DIR  = WORK_DIR / 'Memes_Vibe_Classifier'
GLOVE_PATH = str(WORK_DIR / 'glove.twitter.27B.200d.txt')

# Add project code to Python path
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Override data paths BEFORE any src.* import (read by src/utils/config.py at module load)
os.environ['MMHS_PROJECT_ROOT'] = str(CODE_DIR)   # writable: checkpoints/results/caches go here
os.environ['MMHS_DATA_DIR']     = str(INPUT_DIR)   # read-only: GT json, splits, OCR, images

# Create writable output dirs that would normally sit inside dataset/
(CODE_DIR / 'dataset').mkdir(parents=True, exist_ok=True)   # for token_cache .npy files
(CODE_DIR / 'checkpoints').mkdir(parents=True, exist_ok=True)
(CODE_DIR / 'results').mkdir(parents=True, exist_ok=True)

import torch
print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
print(f"GPUs    : {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {props.name}  ({props.total_memory / 1e9:.1f} GB)")
print(f"\\nCODE_DIR  : {CODE_DIR}")
print(f"INPUT_DIR : {INPUT_DIR}")
"""

CELL_2_CONFIG = """
# Cell 2: Run configuration — EDIT THIS to control what runs

# Which variations to run
RUN_VARIATIONS = ['D']              # Options: 'A', 'B', 'C', 'D'

# Which text modes to run (full ablation = all 3)
TEXT_MODES = ['no_caption', 'tweet_ocr', 'all_text']

# Run only a 1-epoch smoke test on 200-sample subset before full training?
SMOKE_TEST_ONLY = False

# Epoch overrides (None = use defaults from P7Config)
S1_EPOCHS_OVERRIDE = 20            # more data → more epochs before overfitting
S2_EPOCHS_OVERRIDE = 25

DEVICE      = 'auto'
NUM_WORKERS = 4                    # Kaggle: 4 workers (8 caused OOM warnings)
SEED        = 42

MAX_TRAIN_SAMPLES = None           # None = full 150k  ← CRITICAL change from local
MAX_VAL_SAMPLES   = None           # keep full val for reliable metrics

# Batch size: DataParallel across 2 T4s → each GPU sees batch/2
# 2x T4 (15GB each) → batch=256 → 128/GPU in FP16 → ~5GB/GPU ✓
S1_BATCH = 256
S2_BATCH = 128

# LR: 5e-4 base; GloVe embeddings train at 5e-5 (embed_lr_factor=0.1)
S1_LR = 5e-4
S2_LR = 5e-4

print(f"Variations : {RUN_VARIATIONS}")
print(f"Text modes : {TEXT_MODES}")
print(f"Total runs : {len(RUN_VARIATIONS) * len(TEXT_MODES)}")
print(f"Batch S1={S1_BATCH}  S2={S2_BATCH}  workers={NUM_WORKERS}  max_train={MAX_TRAIN_SAMPLES}")
"""

CELL_3_PATHS = """
# Cell 3: Path verification — confirms all required files are present

from pathlib import Path
import os

INPUT_DIR = Path(os.environ['MMHS_DATA_DIR'])
CODE_DIR  = Path(os.environ['MMHS_PROJECT_ROOT'])

REQUIRED_FILES = {
    "GT labels"          : INPUT_DIR / "MMHS150K_GT.json",
    "Processed labels"   : INPUT_DIR / "processed_labels.json",
    "OCR consolidated"   : INPUT_DIR / "ocr_consolidated.json",
    "Train split"        : INPUT_DIR / "splits" / "train_ids.txt",
    "Val split"          : INPUT_DIR / "splits" / "val_ids.txt",
    "Test split"         : INPUT_DIR / "splits" / "test_ids.txt",
    "Images dir"         : INPUT_DIR / "img_resized",
}

all_ok = True
for name, path in REQUIRED_FILES.items():
    exists = path.exists()
    status = "✓" if exists else "✗ MISSING"
    if not exists:
        all_ok = False
    print(f"  {status}  {name}: {path}")

# Count images
img_dir = INPUT_DIR / "img_resized"
if img_dir.exists():
    n_imgs = sum(1 for _ in img_dir.glob("*.jpg"))
    print(f"\\n  Images found: {n_imgs:,}")

# Writable dirs
WRITE_DIRS = {
    "Token cache dir" : CODE_DIR / "dataset",
    "Checkpoints dir" : CODE_DIR / "checkpoints",
    "Results dir"     : CODE_DIR / "results",
}
print()
for name, path in WRITE_DIRS.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"  ✓  {name}: {path}")

if all_ok:
    print("\\n✅ All required files present — ready to train.")
else:
    print("\\n❌ Some files are missing — check your Kaggle dataset attachment.")
"""

CELL_4_TOKENIZER = """
# Cell 4: Tokenizer + token cache build
# The token cache pre-tokenizes all 149k samples once and saves to disk.
# On subsequent runs it loads instantly (~0.5s).

from src.p7.tokenizer import P7Tokenizer
from src.p7.dataset import build_token_cache
from src.p7.config import P7Config

tokenizer = P7Tokenizer(
    model_name="cardiffnlp/twitter-roberta-base",
    max_seq_len=128,
)
print(f"Tokenizer loaded.  Vocab size: {tokenizer.vocab_size:,}")

# Build token cache for all 3 text modes (only builds if cache file is missing)
for tm in ['no_caption', 'tweet_ocr', 'all_text']:
    print(f"  Building token cache: {tm} ...", end=" ")
    build_token_cache(text_mode=tm, max_seq_len=128, tokenizer=tokenizer)
    print("done.")

print("\\nToken caches ready.")
"""

CELL_5_DATASET_SMOKE = """
# Cell 5: Dataset smoke test — verify loading & multi-label labels

from src.p7.config import P7Config
from src.p7.dataset import P7Dataset, HATE_CAT_NAMES, compute_multilabel_from_soft

cfg_test = P7Config(variation='D', text_mode='tweet_ocr')

val_ds = P7Dataset('val', cfg_test, tokenizer, stage=None, is_training=False)

sample = val_ds[0]
print('Sample keys:', list(sample.keys()))
print(f'  image shape:           {sample["image"].shape}')
print(f'  token_ids shape:       {sample["token_ids"].shape}')
print(f'  text (first 80 chars): {sample["text"][:80]}')
print(f'  label_binary:          {sample["label_binary"]}')
print(f'  label_6class:          {sample["label_6class"]}')
print(f'  label_s2:              {sample["label_s2"]}  (-1 = NotHate)')
print(f'  multi_label_binary:    {sample["multi_label_binary"].tolist()}')
print(f'  soft_label:            {[round(x,3) for x in sample["soft_label"].tolist()]}')

# Verify Stage 2 filtering (must contain only hateful samples)
val_s2 = P7Dataset('val', cfg_test, tokenizer, stage=2, is_training=False)
labels_s2 = [val_s2.labels[sid]['hard_label_binary'] for sid in val_s2.sample_ids]
assert all(l == 1 for l in labels_s2), 'Stage 2 filter bug: NotHate samples present!'
print(f'\\nStage 2 filter check: {len(val_s2)} hateful-only samples  ✓')
"""

CELL_6_MODEL_SMOKE = """
# Cell 6: Model smoke test — verify forward pass shapes

import torch
from src.p7.model import MHSDF
from src.p7.config import P7Config

cfg = P7Config(variation='D', text_mode='tweet_ocr')
vocab_size = tokenizer.vocab_size

for nc, label in [(1, 'Stage 1 binary'), (5, 'Stage 2 hate cats'), (6, 'P7-B 6-class')]:
    m = MHSDF.from_config(cfg, vocab_size, num_classes=nc)
    imgs = torch.randn(4, 3, 224, 224)
    toks = torch.randint(0, vocab_size, (4, 128))
    out = m(imgs, toks)
    assert out.shape == (4, nc), f'Expected (4,{nc}), got {out.shape}'
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'  num_classes={nc} ({label}): logits {out.shape}  params={n_params:,}')

# Verify DataParallel wrapping works
import torch.nn as nn
if torch.cuda.device_count() > 1:
    m_dp = nn.DataParallel(m.cuda())
    imgs_gpu = imgs.cuda()
    toks_gpu = toks.cuda()
    out_dp = m_dp(imgs_gpu, toks_gpu)
    print(f'  DataParallel ({torch.cuda.device_count()} GPUs): output {out_dp.shape}  ✓')

print('\\nForward pass check: all shapes correct ✓')
"""

CELL_7_SMOKE_TRAIN = """
# Cell 7: Smoke training test — 1 epoch on 200-sample subset
# Catches integration bugs (AMP, DataParallel, loss, etc.) before full run.

import torch
from torch.utils.data import Subset, DataLoader
from src.p7.dataset import p7_collate_fn
from src.p7.model import MHSDF
from src.p7.losses import get_p7_loss
from src.p7.trainer import resolve_device

device = resolve_device(DEVICE)
cfg_smoke = P7Config(
    variation='D', text_mode='tweet_ocr',
    s1_epochs=1, s2_epochs=1, device=DEVICE,
    num_workers=0, seed=SEED,              # 0 workers for quick smoke test
    use_image_store=False,                 # skip memmap on Kaggle
    use_random_erasing=True,
    token_drop_rate=0.1,
)

train_full = P7Dataset('train', cfg_smoke, tokenizer, stage=1, is_training=True)
smoke_ds   = Subset(train_full, list(range(min(200, len(train_full)))))

class SubsetWrapper(torch.utils.data.Dataset):
    \"\"\"Thin wrapper so loss factory can access .labels and .sample_ids.\"\"\"
    def __init__(self, subset):
        self._ds = subset
        self.labels     = subset.dataset.labels
        self.sample_ids = [subset.dataset.sample_ids[i] for i in subset.indices]
        self.config     = subset.dataset.config
    def __len__(self): return len(self._ds)
    def __getitem__(self, i): return self._ds[i]

smoke_wrapped = SubsetWrapper(smoke_ds)
smoke_loader  = DataLoader(smoke_wrapped, batch_size=16, collate_fn=p7_collate_fn, shuffle=True)

model     = MHSDF.from_config(cfg_smoke, tokenizer.vocab_size, num_classes=1).to(device)
criterion = get_p7_loss('D', 1, smoke_wrapped, device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

model.train()
losses = []
for batch in smoke_loader:
    imgs    = batch['image'].to(device)
    toks    = batch['token_ids'].to(device)
    targets = batch['label_binary'].float().to(device)
    logits  = model(imgs, toks).squeeze(-1)
    loss    = criterion(logits.float(), targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    losses.append(loss.item())

print(f'Smoke test passed ✓  avg_loss={sum(losses)/len(losses):.4f}  n_batches={len(losses)}')
"""

CELL_8_TRAIN = """
# Cell 8: MAIN TRAINING — full ablation across selected variations and text modes

if SMOKE_TEST_ONLY:
    print('SMOKE_TEST_ONLY=True — skipping full training.')
else:
    import logging
    from src.p7.config import P7Config
    from src.p7.trainer import run_p7

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    all_results = {}

    for variation in RUN_VARIATIONS:
        for text_mode in TEXT_MODES:
            run_name = f'p7_{variation}_{text_mode}'
            print(f'\\n{"="*60}')
            print(f'  Starting: {run_name}')
            print(f'{"="*60}')

            cfg = P7Config(
                variation=variation,
                text_mode=text_mode,
                run_name=run_name,
                device=DEVICE,
                num_workers=NUM_WORKERS,
                seed=SEED,

                # ── Dataset ────────────────────────────────────────────
                max_train_samples=MAX_TRAIN_SAMPLES,
                max_val_samples=MAX_VAL_SAMPLES,

                # ── Training ───────────────────────────────────────────
                s1_epochs=S1_EPOCHS_OVERRIDE or 20,
                s2_epochs=S2_EPOCHS_OVERRIDE or 25,
                s1_batch_size=S1_BATCH,
                s2_batch_size=S2_BATCH,
                s1_lr=S1_LR,
                s2_lr=S2_LR,

                # ── Anti-overfitting ───────────────────────────────────
                early_stop_patience=5,
                label_smoothing=0.1,
                weight_decay=5e-4,
                s1_loss='focal',
                use_random_erasing=True,
                token_drop_rate=0.10,

                # ── LR scheduling ─────────────────────────────────────
                embed_lr_factor=0.1,           # GloVe embeds at 10× lower LR
                warmup_epochs=1,               # 1-epoch linear warmup

                # ── Hardware ───────────────────────────────────────────
                use_amp=True,
                use_data_parallel=True,        # uses both T4 GPUs automatically
                use_image_store=False,          # disabled: Kaggle storage limited
                use_compile=False,
                grad_accum_steps=2,             # 2 micro-steps → effective batch=256, peak VRAM=128/GPU

                # ── GloVe Twitter embeddings ───────────────────────────
                use_glove=True,
                glove_path=GLOVE_PATH,
                glove_dim=200,
                embed_dim=200,                  # must match glove_dim
            )

            metrics = run_p7(cfg)
            all_results[run_name] = metrics
            print(f'\\n  ✓ Done: {run_name}')

    print(f'\\n✓ All {len(all_results)} runs complete!')
"""

CELL_9_RESULTS = """
# Cell 9: Results comparison table

from pathlib import Path
import json, os

CODE_DIR    = Path(os.environ['MMHS_PROJECT_ROOT'])
results_base = CODE_DIR / 'results' / 'p7'

rows = []
for run_dir in sorted(results_base.iterdir()) if results_base.exists() else []:
    mfile = run_dir / 'metrics.json'
    if not mfile.exists():
        continue
    with open(mfile) as f:
        m = json.load(f)

    variation = m.get('variation', '?')
    run_name  = m.get('config', run_dir.name)
    s1_f1     = m.get('stage1', {}).get('best', None)
    s2_key    = 'stage2_calibrated' if 'stage2_calibrated' in m else 'stage2'
    s2_f1     = (m.get(s2_key, {}).get('best', None)
                 or m.get(s2_key, {}).get('multilabel/macro_f1', None))
    composite = m.get('pipeline', {}).get('pipeline/composite', None)

    rows.append({
        'run'       : run_name,
        'var'       : variation,
        's1_f1'     : f'{s1_f1:.4f}' if s1_f1 else '—',
        's2_f1'     : f'{s2_f1:.4f}' if s2_f1 else '—',
        'composite' : f'{composite:.4f}' if composite else '—',
    })

if rows:
    header = f'{"Run":<30} {"Var":<5} {"S1 Macro F1":<14} {"S2 Macro F1":<14} {"Composite":<12}'
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f'{r["run"]:<30} {r["var"]:<5} {r["s1_f1"]:<14} {r["s2_f1"]:<14} {r["composite"]:<12}')
else:
    print('No completed runs found yet.')
"""

CELL_10_TEST = """
# Cell 10: Test-set evaluation (run after Cell 8 completes)

if SMOKE_TEST_ONLY:
    print('Skipping test eval in smoke-test mode.')
else:
    import torch, numpy as np, os
    from pathlib import Path
    from torch.utils.data import DataLoader
    from src.p7.config import P7Config
    from src.p7.dataset import P7Dataset, p7_collate_fn
    from src.p7.model import MHSDF
    from src.p7.metrics import apply_thresholds, compute_multilabel_metrics, HATE_CAT_NAMES
    from src.p7.trainer import resolve_device
    from src.evaluation.metrics import compute_binary_metrics

    CODE_DIR    = Path(os.environ['MMHS_PROJECT_ROOT'])
    PRIMARY_RUN = 'p7_D_tweet_ocr'
    ckpt_dir    = CODE_DIR / 'checkpoints' / 'p7' / PRIMARY_RUN

    cfg    = P7Config(variation='D', text_mode='tweet_ocr',
                      use_image_store=False, embed_dim=200)
    device = resolve_device(cfg.device)

    from src.p7.tokenizer import P7Tokenizer
    tok = P7Tokenizer(cfg.bert_model_name, cfg.max_seq_len)

    # ── Stage 1 ────────────────────────────────────────────────────────────────
    test_s1   = P7Dataset('test', cfg, tok, stage=1, is_training=False)
    s1_loader = DataLoader(test_s1, batch_size=256, collate_fn=p7_collate_fn,
                           num_workers=4, pin_memory=True)

    model_s1 = MHSDF.from_config(cfg, tok.vocab_size, num_classes=1).to(device)
    model_s1.load_state_dict(torch.load(ckpt_dir / 'stage1_best.pt', map_location=device))
    model_s1.eval()

    all_logits_s1, all_targets_s1 = [], []
    with torch.no_grad():
        for batch in s1_loader:
            logits = model_s1(batch['image'].to(device), batch['token_ids'].to(device))
            all_logits_s1.append(logits.cpu())
            all_targets_s1.append(batch['label_binary'])

    logits_s1  = torch.cat(all_logits_s1).squeeze(-1).numpy()
    targets_s1 = torch.cat(all_targets_s1).numpy()
    probs_s1   = 1 / (1 + np.exp(-logits_s1))
    preds_s1   = (probs_s1 >= 0.5).astype(int)

    s1_metrics = compute_binary_metrics(targets_s1, preds_s1, probs_s1)
    print('=== Stage 1 (Binary) — TEST ===')
    for k, v in s1_metrics.items():
        print(f'  {k}: {v:.4f}')

    # ── Stage 2 ────────────────────────────────────────────────────────────────
    test_s2   = P7Dataset('test', cfg, tok, stage=2, is_training=False)
    s2_loader = DataLoader(test_s2, batch_size=256, collate_fn=p7_collate_fn,
                           num_workers=4, pin_memory=True)

    model_s2 = MHSDF.from_config(cfg, tok.vocab_size, num_classes=5).to(device)
    model_s2.load_state_dict(torch.load(ckpt_dir / 'stage2_best.pt', map_location=device))
    model_s2.eval()

    thresholds        = np.load(ckpt_dir / 'stage2_thresholds.npy')
    all_logits_s2, all_targets_s2 = [], []
    with torch.no_grad():
        for batch in s2_loader:
            logits = model_s2(batch['image'].to(device), batch['token_ids'].to(device))
            all_logits_s2.append(logits.cpu())
            all_targets_s2.append(batch['multi_label_binary'][:, 1:])

    logits_s2  = torch.cat(all_logits_s2).numpy()
    targets_s2 = torch.cat(all_targets_s2).numpy()
    preds_s2   = apply_thresholds(logits_s2, thresholds)

    s2_metrics = compute_multilabel_metrics(targets_s2.astype(int), preds_s2, HATE_CAT_NAMES)
    print('\\n=== Stage 2 (Multilabel) — TEST ===')
    for k, v in s2_metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
"""

# ── Assemble notebook ─────────────────────────────────────────────────────────

cells = [
    md("""# P7 — MHSDF Pipeline (Kaggle 2×T4)

**CNN + BiLSTM multimodal hate speech classifier** — full ablation across 4 variations × 3 text modes.

## Quick-start
1. **Run Cell 0** (install deps + download GloVe) — internet must be ON in Kaggle settings
2. **Run Cell 1** (env setup) — sets paths before any import
3. **Run Cell 2** (config) — adjust variations/text modes as needed
4. **Run Cells 3–7** (verification + smoke tests)
5. **Run Cell 8** (main training) — 12 runs, ~8–12h on 2×T4

## Setup checklist
- ✅ Dataset attached: `ekanshkhullar/updated-hate-speech-dataset`
- ✅ Internet: ON (for GloVe download + HuggingFace tokenizer)
- ✅ Accelerator: GPU × 2 (T4 ×2)
- ✅ Code at: `/kaggle/working/Memes_Vibe_Classifier/`
"""),
    code(CELL_0_INSTALL),
    code(CELL_1_ENV),
    code(CELL_2_CONFIG),
    code(CELL_3_PATHS),
    code(CELL_4_TOKENIZER),
    code(CELL_5_DATASET_SMOKE),
    code(CELL_6_MODEL_SMOKE),
    code(CELL_7_SMOKE_TRAIN),
    code(CELL_8_TRAIN),
    code(CELL_9_RESULTS),
    code(CELL_10_TEST),
]

nb = {
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
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {OUT}")
print(f"Cells: {len(cells)}")
