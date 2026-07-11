#!/usr/bin/env bash
# Download MMHS dataset from Kaggle onto the remote A6000 machine.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python scripts/setup_kaggle_data.py "$@"
