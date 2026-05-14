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
S1_EPOCHS     = 5
S1_LR         = 2e-4
S1_BATCH_SIZE = 16         # T4-safe; effective batch = 16 × 8 = 128
S1_WARMUP     = 0.05       # 5% of total steps

# ── STAGE 2 HYPERPARAMETERS (C/D only) ──────────────────────────────────────
S2_EPOCHS     = 7
S2_LR         = 1e-4
S2_BATCH_SIZE = 16
S2_WARMUP     = 0.10       # 10% of total steps

# ── SHARED ───────────────────────────────────────────────────────────────────
GRAD_ACCUM    = 8          # effective batch = batch_size × grad_accum
SEED          = 42
USE_AMP       = True       # fp16 mixed precision
USE_DATA_PARALLEL = True   # use both T4 GPUs

print(f"Variation : P2-{VARIATION}")
print(f"Text mode : {TEXT_MODE}")
print(f"Eff batch : {S1_BATCH_SIZE} × {GRAD_ACCUM} = {S1_BATCH_SIZE * GRAD_ACCUM}")
print(f"Max train : {MAX_TRAIN_SAMPLES or 'full dataset'}")"""))

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
    # ── Kaggle paths (adjust dataset name to match your upload) ──────────────
    DATASET_NAME = "mmhs150k-processed"   # <-- change to your Kaggle dataset name
    INPUT_DIR    = Path(f"/kaggle/input/{DATASET_NAME}")
    OUTPUT_DIR   = Path("/kaggle/working")
    
    os.environ["MMHS_PROJECT_ROOT"] = str(OUTPUT_DIR)
    os.environ["MMHS_DATA_DIR"]     = str(INPUT_DIR)
    
    # Add src to path
    SRC_DIR = INPUT_DIR / "src"  # adjust if repo uploaded differently
    if SRC_DIR.exists():
        sys.path.insert(0, str(INPUT_DIR))
    else:
        # Repo uploaded as separate dataset
        REPO_DIR = Path("/kaggle/input/mmhs150k-repo")
        if REPO_DIR.exists():
            sys.path.insert(0, str(REPO_DIR))
else:
    # ── Local paths ───────────────────────────────────────────────────────────
    PROJECT_ROOT = Path.cwd()
    while not (PROJECT_ROOT / "src").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Project root : {PROJECT_ROOT}")

# Verify imports work
try:
    from src.p2.config import P2Config
    print("✓ P2 imports OK")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Check that project src/ is on sys.path")"""))

# ── Cell 5: Build config ──────────────────────────────────────────────────────
cells.append(md("## 🔧 5. Build Experiment Config"))
cells.append(cell("""from src.p2.config import P2Config

config = P2Config(
    variation         = VARIATION,
    text_mode         = TEXT_MODE,
    s1_epochs         = S1_EPOCHS,
    s1_lr             = S1_LR,
    s1_batch_size     = S1_BATCH_SIZE,
    s1_warmup_ratio   = S1_WARMUP,
    s2_epochs         = S2_EPOCHS,
    s2_lr             = S2_LR,
    s2_batch_size     = S2_BATCH_SIZE,
    s2_warmup_ratio   = S2_WARMUP,
    grad_accum_steps  = GRAD_ACCUM,
    seed              = SEED,
    use_amp           = USE_AMP,
    use_data_parallel = USE_DATA_PARALLEL,
    max_train_samples = MAX_TRAIN_SAMPLES,
    max_val_samples   = MAX_VAL_SAMPLES,
    device            = "auto",
    num_workers       = 2,   # Kaggle safe default
)

print(f"Run name     : {config.run_name}")
print(f"Checkpoint   : {config.run_dir}")
print(f"Results      : {config.results_run_dir}")
print(f"Eff batch    : {config.s1_batch_size} × {config.grad_accum_steps} = {config.s1_batch_size * config.grad_accum_steps}")"""))

# ── Cell 6: Smoke test ────────────────────────────────────────────────────────
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

# Verify proj_t identity init
import torch.nn.functional as F
eye = torch.eye(768)
is_identity = F.mse_loss(model.proj_t.weight.data.cpu(), eye).item()
print(f"  proj_t identity : MSE={is_identity:.2e}  (expect ~0.0)")

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
print(f"  Patch tokens : {V.shape}  (expect [2, 257, 768])")

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
