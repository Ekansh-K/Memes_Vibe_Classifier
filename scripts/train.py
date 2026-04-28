"""Main training script (Phase 1+ — stub for now)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ExperimentConfig
from src.utils.logging_utils import set_seed


def main():
    # Placeholder — implemented in Phase 1
    raise NotImplementedError("Training script will be implemented in Phase 1")


if __name__ == "__main__":
    main()
