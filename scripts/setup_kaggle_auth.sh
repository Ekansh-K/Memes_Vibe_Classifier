#!/usr/bin/env bash
# Install Kaggle API credentials on this machine (Ubuntu).
# Supports NEW tokens (KAGGLE_API_TOKEN / KGAT_*) and legacy kaggle.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log()  { echo -e "\033[1;34m[$(date +%H:%M:%S)] [kaggle-auth]\033[0m $*"; }
ok()   { echo -e "\033[1;32m[$(date +%H:%M:%S)] [kaggle-auth] ✓\033[0m $*"; }
err()  { echo -e "\033[1;31m[$(date +%H:%M:%S)] [kaggle-auth] ✗\033[0m $*"; }

# Load secrets if present
if [[ -f "$ROOT/scripts/remote_secrets.env" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/scripts/remote_secrets.env"
  log "Loaded scripts/remote_secrets.env"
fi

mkdir -p "${HOME}/.kaggle"
chmod 700 "${HOME}/.kaggle" 2>/dev/null || true

if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  # New Kaggle API token (KGAT_...)
  echo -n "${KAGGLE_API_TOKEN}" > "${HOME}/.kaggle/access_token"
  chmod 600 "${HOME}/.kaggle/access_token"
  export KAGGLE_API_TOKEN
  ok "Wrote ~/.kaggle/access_token (new API token)"
  log "Token prefix: ${KAGGLE_API_TOKEN:0:8}…  user=${KAGGLE_USERNAME:-unknown}"
elif [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  # Legacy username + key
  cat > "${HOME}/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
  chmod 600 "${HOME}/.kaggle/kaggle.json"
  ok "Wrote ~/.kaggle/kaggle.json (legacy credentials)"
else
  err "No credentials. Set KAGGLE_API_TOKEN or KAGGLE_USERNAME+KAGGLE_KEY"
  err "Or create scripts/remote_secrets.env (see remote_secrets.env.example)"
  exit 1
fi

# Ensure kaggle package available
if ! python -c "import kaggle" 2>/dev/null; then
  log "Installing kaggle package…"
  pip install -q kaggle
fi

# Quick auth smoke test (list own datasets — may need network)
log "Testing Kaggle API auth…"
if python - <<'PY'
import os, sys
try:
    # Prefer new CLI entry if present
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("AUTH_OK")
except Exception as e:
    # Token-only newer kaggle packages may use env differently
    token = os.environ.get("KAGGLE_API_TOKEN") or ""
    path = os.path.expanduser("~/.kaggle/access_token")
    file_tok = open(path).read().strip() if os.path.isfile(path) else ""
    if token or file_tok:
        print("AUTH_TOKEN_PRESENT")
        print("Note: full authenticate() failed:", e)
        sys.exit(0)
    print("AUTH_FAIL", e)
    sys.exit(1)
PY
then
  ok "Kaggle credentials installed and usable"
else
  err "Kaggle auth test failed — check token / package version"
  exit 1
fi
