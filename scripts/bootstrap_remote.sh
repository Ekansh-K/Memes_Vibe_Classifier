#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# ONE-SHOT BOOTSTRAP for Ubuntu 22.04 + CUDA 12.x + RTX A6000
#
# Run from repo root AFTER: git clone … && cd EndSem_Project
#
#   bash scripts/bootstrap_remote.sh              # setup + data + smoke
#   bash scripts/bootstrap_remote.sh --full       # setup + data + full Stage-1
#   bash scripts/bootstrap_remote.sh --setup-only # env + data only (no train)
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="smoke"   # smoke | full | setup-only
for arg in "$@"; do
  case "$arg" in
    --full) MODE="full" ;;
    --setup-only) MODE="setup-only" ;;
    --smoke) MODE="smoke" ;;
  esac
done

log()  { echo -e "\033[1;34m[$(date +%H:%M:%S)] [bootstrap]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[$(date +%H:%M:%S)] [bootstrap] ✓\033[0m $*"; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] [bootstrap] ✗\033[0m $*"; }
section() {
  echo ""
  echo -e "\033[1;35m════════════════════════════════════════════════════════════\033[0m"
  echo -e "\033[1;35m  $*\033[0m"
  echo -e "\033[1;35m════════════════════════════════════════════════════════════\033[0m"
}

section "0) Environment"
log "Host: $(hostname)  User: $(whoami)  PWD: $ROOT"
log "Mode: $MODE"
log "Date: $(date -Is)"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  err "nvidia-smi not found — GPU may be unavailable"
fi

# Load secrets (create from env if file missing but token already exported)
if [[ ! -f "$ROOT/scripts/remote_secrets.env" ]]; then
  if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
    log "Creating scripts/remote_secrets.env from current environment…"
    cat > "$ROOT/scripts/remote_secrets.env" <<EOF
export KAGGLE_USERNAME="${KAGGLE_USERNAME:-ekanshkhullar}"
export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN}"
export KAGGLE_DATASET_SLUG="${KAGGLE_DATASET_SLUG:-ekanshkhullar/updated-hate-speech-dataset}"
EOF
    chmod 600 "$ROOT/scripts/remote_secrets.env"
  else
    err "Missing scripts/remote_secrets.env and KAGGLE_API_TOKEN not set"
    err "Create the file (see docs/REMOTE_BOOT_COMMANDS.md) then re-run"
    exit 1
  fi
fi
# shellcheck disable=SC1091
source "$ROOT/scripts/remote_secrets.env"
ok "Loaded remote_secrets.env (user=${KAGGLE_USERNAME:-?} token_set=${KAGGLE_API_TOKEN:+yes})"

export MMHS_PROJECT_ROOT="${MMHS_PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

section "1) Python venv"
if [[ ! -d "$ROOT/.venv" ]]; then
  log "Creating .venv…"
  python3 -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
ok "Python: $(python --version) @ $(which python)"

log "Upgrading pip…"
pip install -U pip setuptools wheel -q

section "2) PyTorch (CUDA 12.x wheels — works with toolkit 12.9)"
# cu124 wheels are compatible with driver stacks for CUDA 12.x
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  ok "torch already installed with CUDA: $(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))')"
else
  log "Installing torch+torchvision (cu124 index)…"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

section "3) Project requirements"
log "pip install -r requirements.txt …"
pip install -r requirements.txt
pip install -q peft accelerate kaggle
# OpenAI CLIP package used by some legacy paths / P2
pip install -q "git+https://github.com/openai/CLIP.git" || log "CLIP git install skipped/failed (HF CLIP still works for HateCLIPper)"

python - <<'PY'
import torch
print(f"[bootstrap] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[bootstrap] GPU={torch.cuda.get_device_name(0)}  VRAM={torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
else:
    raise SystemExit("CUDA not available in PyTorch — check driver / wheel")
PY
ok "GPU stack ready"

section "4) Kaggle auth"
bash "$ROOT/scripts/setup_kaggle_auth.sh"

section "5) Download dataset (if needed)"
python "$ROOT/scripts/setup_kaggle_data.py"
# Always re-verify explicitly with progress
python "$ROOT/scripts/verify_dataset.py"

section "6) Captions check"
if [[ -f "$ROOT/results/vlm_captions.json" ]]; then
  ok "Found results/vlm_captions.json"
elif [[ -f "$ROOT/dataset/vlm_captions.json" ]]; then
  ok "Found dataset/vlm_captions.json"
else
  log "! No vlm_captions.json — Stage-1 will fall back to tweet+OCR for all_text"
  log "  Optional: scp results/vlm_captions.json from your laptop into results/"
fi

mkdir -p "$ROOT/checkpoints/stage1" "$ROOT/results/stage1" "$ROOT/logs"

if [[ "$MODE" == "setup-only" ]]; then
  section "DONE (setup-only)"
  ok "Environment + dataset ready. Next:"
  echo "  source .venv/bin/activate"
  echo "  bash scripts/run_all_stage1.sh --smoke"
  echo "  bash scripts/run_all_stage1.sh"
  exit 0
fi

section "7) Stage-1 training"
if [[ "$MODE" == "smoke" ]]; then
  log "Starting SMOKE Stage-1 (small subsample)…"
  bash "$ROOT/scripts/run_all_stage1.sh" --smoke 2>&1 | tee "$ROOT/logs/stage1_smoke_$(date +%Y%m%d_%H%M%S).log"
else
  log "Starting FULL Stage-1 stack…"
  bash "$ROOT/scripts/run_all_stage1.sh" 2>&1 | tee "$ROOT/logs/stage1_full_$(date +%Y%m%d_%H%M%S).log"
fi

section "DONE"
ok "Logs under logs/   metrics under results/stage1/"
ls -la "$ROOT/results/stage1/" 2>/dev/null || true
if [[ -f "$ROOT/results/stage1/ensemble/metrics.json" ]]; then
  ok "Ensemble metrics:"
  python -m json.tool "$ROOT/results/stage1/ensemble/metrics.json" | head -40
fi
