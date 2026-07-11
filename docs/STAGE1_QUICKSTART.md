# Stage-1 Quickstart (Remote RTX A6000 48GB)

## What was built

Focus: **fix Stage-1 hate/not-hate** (the bottleneck). Stage-2 is already strong.

| Component | Path | What it does |
|---|---|---|
| Shared data + eval | `src/stage1/` | Soft labels, captions, threshold sweep, Macro F1 / Hate Recall / AUC |
| Text full fine-tune | `src/s1_text/` | HateBERT / twitter-roberta-hate / DeBERTa — **highest ROI** |
| Hate-CLIPper + adapters | `src/hateclipper_mmhs/` | Port of EMNLP Hate-CLIPper + MemeCLIP residual adapters |
| Qwen3-VL LoRA | `src/s1_vlm/` | **Qwen3-VL** (not 2.5) generative Stage-1 |
| Ensemble | `scripts/run_s1_ensemble.py` | Average val probs → best threshold |
| Master script | `scripts/run_all_stage1.sh` / `.ps1` | Full stack one command |
| Kaggle data | `scripts/setup_kaggle_data.py` | `ekanshkhullar/updated-hate-speech-dataset` |

Reference clones: `third_party/hateclipper`, `third_party/MemeCLIP` (logic ported into `src/`).

P2 TCAM default now **unfreezes last 4 TweetEval layers** (`src/p2/config.py`).

## On the A6000 machine

```bash
git clone <this-repo> && cd EndSem_Project
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git   # optional if using openai clip elsewhere
pip install peft accelerate

# Kaggle API: ~/.kaggle/kaggle.json  OR  export KAGGLE_USERNAME + KAGGLE_KEY
python scripts/setup_kaggle_data.py

# Copy captions if not in the zip (from local project):
# scp results/vlm_captions.json remote:EndSem_Project/results/

# Full Stage-1 stack
bash scripts/run_all_stage1.sh

# Smoke test first (recommended once)
bash scripts/run_all_stage1.sh --smoke

# Skip VLM if short on time
bash scripts/run_all_stage1.sh --skip-vlm
```

Windows remote:

```powershell
.\scripts\run_all_stage1.ps1
.\scripts\run_all_stage1.ps1 -Smoke
.\scripts\run_all_stage1.ps1 -SkipVlm
```

## Metrics to watch

- **Macro F1**, **Hate F1**, **Hate Recall**, **AUC-ROC**
- Accuracy alone is misleading (majority ~82.8%)

Outputs: `results/stage1/<run>/metrics.json` + `val_preds.npz`  
Ensemble: `results/stage1/ensemble/metrics.json`  
Checkpoints: `checkpoints/stage1/<run>/best.pt`
