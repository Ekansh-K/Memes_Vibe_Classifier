# Remote A6000 Runbook — Stage-1 Improvement Stack

## 1. One-time setup on the cloud machine

```bash
# Clone your repo
git clone <YOUR_REPO_URL> EndSem_Project
cd EndSem_Project

# Python env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
pip install peft accelerate bitsandbytes  # for Qwen3-VL LoRA
# optional kaggle CLI
pip install kaggle

# Dataset from Kaggle
export KAGGLE_USERNAME=your_user
export KAGGLE_KEY=your_key
bash scripts/setup_kaggle_data.sh
# Verify:
ls dataset/MMHS150K_GT.json dataset/processed_labels.json dataset/img_resized | head
```

## 2. Run full Stage-1 pipeline (recommended)

```bash
# Full optimized Stage-1 stack (text + HateCLIPper + ensemble)
bash scripts/run_all_stage1.sh

# Smoke test first (optional)
bash scripts/run_all_stage1.sh --smoke
```

## 3. Individual stages

```bash
# Text super-baseline (highest ROI)
python scripts/run_s1_text.py --model twitter-roberta --text_mode all_text

# Hate-specialized text
python scripts/run_s1_text.py --model hate-latest --text_mode all_text

# DeBERTa-large (heavier)
python scripts/run_s1_text.py --model deberta-large --text_mode all_text --batch_size 32

# Hate-CLIPper align fusion (multimodal)
python scripts/run_s1_hateclipper.py --fusion align --use_adapters

# Qwen3-VL LoRA Stage-1 (newer VLM, not 2.5)
python scripts/run_s1_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --epochs 2

# Ensemble val predictions → best threshold
python scripts/run_s1_ensemble.py

# Optional: upgraded P2 TCAM with unfrozen text
python scripts/run_p2.py --variation D --text_mode all_text --s1_epochs 8
```

## 4. Outputs

```
checkpoints/stage1/<run_name>/best.pt
results/stage1/<run_name>/metrics.json
results/stage1/<run_name>/val_preds.npz   # for ensemble
results/stage1/ensemble/metrics.json
```

## 5. Metrics that matter

Primary: **Macro F1**, **Hate F1**, **Hate Recall**, **AUC-ROC**  
Secondary: Accuracy (majority baseline ≈ 82.8% — do not optimize for this alone)
