# Final boot commands — Ubuntu 22.04 + CUDA 12.9 + A6000

Credentials are pre-filled in `scripts/remote_secrets.env` (gitignored):

- **Kaggle user:** `ekanshkhullar`
- **Token name:** `dataset`
- **Token:** `KAGGLE_API_TOKEN=KGAT_…` (new API style)
- **Dataset:** `ekanshkhullar/updated-hate-speech-dataset`

Dataset target: **`./dataset/`** under the repo (auto-download + auto-verify).

---

## Copy-paste on first boot

```bash
# ── 1. Clone repo ──────────────────────────────────────────
git clone <YOUR_REPO_URL> EndSem_Project
cd EndSem_Project

# If remote_secrets.env was NOT pushed (gitignored), create it:
# nano scripts/remote_secrets.env
# (paste contents from your local machine)

# ── 2. Make scripts executable ─────────────────────────────
chmod +x scripts/*.sh

# ── 3. ONE command: install stack + download data + smoke train
bash scripts/bootstrap_remote.sh
```

### Modes

```bash
bash scripts/bootstrap_remote.sh --setup-only   # env + dataset only, no train
bash scripts/bootstrap_remote.sh --smoke        # default: small Stage-1 run
bash scripts/bootstrap_remote.sh --full         # full Stage-1 stack (hours–days)
```

### After setup-only, train yourself

```bash
cd EndSem_Project
source .venv/bin/activate
source scripts/remote_secrets.env

# Verify anytime
python scripts/verify_dataset.py

# Smoke Stage-1
bash scripts/run_all_stage1.sh --smoke

# Full Stage-1 (text + HateCLIPper + Qwen3-VL LoRA + ensemble)
bash scripts/run_all_stage1.sh

# Full without VLM
bash scripts/run_all_stage1.sh --skip-vlm
```

---

## What each script does

| Script | Purpose |
|---|---|
| `scripts/remote_secrets.env` | Your Kaggle user + API token |
| `scripts/setup_kaggle_auth.sh` | Writes `~/.kaggle/access_token` |
| `scripts/setup_kaggle_data.py` | Downloads Kaggle dataset → `./dataset/`, then verifies |
| `scripts/verify_dataset.py` | Checks all required files, splits, sample images, label schema |
| `scripts/bootstrap_remote.sh` | Full first-boot: venv, torch CUDA, deps, auth, data, train |
| `scripts/run_all_stage1.sh` | Stage-1 stack only (assumes env ready) |

---

## Security

`remote_secrets.env` is **gitignored**. If this token was shared in chat or committed by mistake, **rotate it** on Kaggle Settings → API.
