#!/usr/bin/env bash
# Master Stage-1 pipeline for RTX A6000 (48GB) on Ubuntu.
# Usage:
#   bash scripts/run_all_stage1.sh
#   bash scripts/run_all_stage1.sh --smoke
#   bash scripts/run_all_stage1.sh --skip-vlm
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Load secrets + unbuffered logs
if [[ -f "$ROOT/scripts/remote_secrets.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/remote_secrets.env"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MMHS_PROJECT_ROOT="${MMHS_PROJECT_ROOT:-$ROOT}"

# Activate venv if present
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

SMOKE=0
SKIP_VLM=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
    --skip-vlm) SKIP_VLM=1 ;;
  esac
done

log()  { echo -e "\033[1;34m[$(date +%H:%M:%S)] [stage1]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[$(date +%H:%M:%S)] [stage1] ✓\033[0m $*"; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] [stage1] ✗\033[0m $*"; }
section() {
  echo ""
  echo -e "\033[1;36m────────────────────────────────────────────────────────────\033[0m"
  echo -e "\033[1;36m  $*\033[0m"
  echo -e "\033[1;36m────────────────────────────────────────────────────────────\033[0m"
}

EXTRA=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA+=(--max_train_samples 2000 --max_val_samples 500)
  EPOCHS_TEXT=1
  EPOCHS_HC=2
  EPOCHS_VLM=1
else
  EPOCHS_TEXT=4
  EPOCHS_HC=12
  EPOCHS_VLM=2
fi

mkdir -p "$ROOT/logs" "$ROOT/checkpoints/stage1" "$ROOT/results/stage1"
LOG_FILE="$ROOT/logs/run_all_stage1_$(date +%Y%m%d_%H%M%S).log"
# Tee everything
exec > >(tee -a "$LOG_FILE") 2>&1

section "MMHS Stage-1 | smoke=$SMOKE skip_vlm=$SKIP_VLM"
log "ROOT=$ROOT"
log "Log file: $LOG_FILE"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
fi

# 0) Data
section "0/5 Dataset check + download if needed"
if [[ ! -f dataset/MMHS150K_GT.json ]]; then
  log "dataset/MMHS150K_GT.json missing — downloading from Kaggle…"
  bash "$ROOT/scripts/setup_kaggle_auth.sh" || true
  python scripts/setup_kaggle_data.py
else
  ok "dataset/MMHS150K_GT.json present"
fi
log "Verifying dataset files…"
python scripts/verify_dataset.py || {
  err "Dataset verification failed"
  exit 1
}
ok "Dataset verified"

# 1) Text models
section "1/5 Text Stage-1: hate-latest (highest ROI)"
log "Start: run_s1_text.py hate-latest epochs=$EPOCHS_TEXT"
python scripts/run_s1_text.py --model hate-latest --text_mode all_text --epochs "$EPOCHS_TEXT" "${EXTRA[@]:-}"
ok "hate-latest finished → results/stage1/"

section "2/5 Text Stage-1: twitter-roberta"
log "Start: run_s1_text.py twitter-roberta epochs=$EPOCHS_TEXT"
python scripts/run_s1_text.py --model twitter-roberta --text_mode all_text --epochs "$EPOCHS_TEXT" "${EXTRA[@]:-}"
ok "twitter-roberta finished"

if [[ "$SMOKE" != "1" ]]; then
  section "2b/5 Text Stage-1: deberta-large (optional)"
  log "Start: deberta-large (soft-fail on OOM)"
  python scripts/run_s1_text.py --model deberta-large --text_mode all_text --epochs 3 --batch_size 32 "${EXTRA[@]:-}" \
    || log "DeBERTa failed/skipped — continuing stack"
fi

# 2) Hate-CLIPper
section "3/5 Hate-CLIPper multimodal (align + adapters)"
log "Start: run_s1_hateclipper.py epochs=$EPOCHS_HC"
python scripts/run_s1_hateclipper.py --fusion align --epochs "$EPOCHS_HC" "${EXTRA[@]:-}"
ok "Hate-CLIPper finished"

# 3) Qwen3-VL LoRA
if [[ "$SKIP_VLM" != "1" ]]; then
  section "4/5 Qwen3-VL LoRA Stage-1 fine-tune"
  log "Start: run_s1_vlm.py Qwen3-VL-8B epochs=$EPOCHS_VLM"
  log "This is the slowest step (may take many hours)."
  python scripts/run_s1_vlm.py --model "Qwen/Qwen3-VL-8B-Instruct" --epochs "$EPOCHS_VLM" "${EXTRA[@]:-}" \
    || log "VLM failed — ensemble will use text + HateCLIPper only"
else
  section "4/5 VLM skipped (--skip-vlm)"
fi

# 4) Ensemble
section "5/5 Ensemble Stage-1 predictions"
log "Start: run_s1_ensemble.py"
python scripts/run_s1_ensemble.py
ok "Ensemble done"

section "COMPLETE"
ok "Per-run metrics: results/stage1/*/metrics.json"
ok "Ensemble:        results/stage1/ensemble/metrics.json"
ok "Checkpoints:     checkpoints/stage1/"
ok "This log:        $LOG_FILE"
if [[ -f results/stage1/ensemble/metrics.json ]]; then
  python -m json.tool results/stage1/ensemble/metrics.json | head -50
fi
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi
