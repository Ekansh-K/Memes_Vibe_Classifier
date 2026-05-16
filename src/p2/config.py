"""P2Config — configuration dataclass for all P2 (TCAM) variations.

Variations:
    A — Binary-only (Stage 1 standalone)
    B — Direct 6-class multi-class (single stage)
    C — Two-stage hierarchical, single-label Stage 2
    D — Two-stage hierarchical, multi-label Stage 2 (primary)

Text modes (ablation):
    tweet_ocr  — tweet_text [SEP] ocr_text
    all_text   — caption [SEP] ocr_text [SEP] tweet_text
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.utils.config import PROJECT_ROOT


@dataclass
class P2Config:
    """Full configuration for a single P2 experiment run."""

    # ── Experiment identity ─────────────────────────────────────────────────
    variation: Literal["A", "B", "C", "D"] = "D"
    text_mode: Literal["tweet_ocr", "all_text"] = "all_text"
    run_name: str = "p2_D_all_text"

    # ── Backbone models (frozen) ────────────────────────────────────────────
    clip_model: str = "ViT-L/14"
    tweet_model: str = "cardiffnlp/twitter-roberta-base"

    # ── Architecture ────────────────────────────────────────────────────────
    d_v: int = 1024           # CLIP ViT-L/14 visual patch embedding dim
    d_t: int = 768            # TweetEval RoBERTa hidden dim
    n_heads: int = 8              # cross-attention heads (must divide d_v)
    head_hidden: int = 512        # classification head hidden dim
    head_dropout: float = 0.3     # dropout in classification head
    max_text_len: int = 128       # TweetEval tokenizer max length

    # ── Stage 1 training ────────────────────────────────────────────────────
    s1_epochs: int = 5
    s1_lr: float = 2e-4
    s1_batch_size: int = 16       # T4-safe (effective batch = 16 × 8 = 128)
    s1_warmup_ratio: float = 0.05

    # ── Stage 2 training ────────────────────────────────────────────────────
    s2_epochs: int = 7
    s2_lr: float = 1e-4
    s2_batch_size: int = 16       # T4-safe
    s2_warmup_ratio: float = 0.10

    # ── Shared training ─────────────────────────────────────────────────────
    weight_decay: float = 0.01
    grad_accum_steps: int = 8     # effective batch = 16 × 8 = 128
    max_grad_norm: float = 1.0
    scheduler: str = "cosine"
    use_amp: bool = True          # fp16
    early_stop_patience: int = 3  # epochs without val improvement → stop
    label_smoothing: float = 0.0

    # ── Hardware ────────────────────────────────────────────────────────────
    device: str = "auto"          # "auto" → cuda if available, else cpu
    use_data_parallel: bool = True  # nn.DataParallel across all visible GPUs
    num_workers: int = 4

    # ── Data ────────────────────────────────────────────────────────────────
    img_size: int = 224
    ocr_source: str = "filtered"      # "filtered" (clean), "new" (raw), "old", or "both"
    multilabel_threshold: float = 2 / 3  # soft_label[c] >= 2/3 → active
    use_image_store: bool = False  # keep False on Kaggle (disk space limited)
    exclude_full_disagreement: bool = False

    # ── Dataset size control ────────────────────────────────────────────────
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    # ── Paths ───────────────────────────────────────────────────────────────
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "p2")
    results_dir: str = str(PROJECT_ROOT / "results" / "p2")

    # ── Misc ────────────────────────────────────────────────────────────────
    seed: int = 42
    log_every_n_steps: int = 50

    # ── Derived properties ──────────────────────────────────────────────────
    @property
    def num_classes_s1(self) -> int:
        return 1

    @property
    def num_classes_s2(self) -> int:
        """Stage 2 predicts over 5 hate categories (excludes NotHate=0)."""
        return 5

    @property
    def num_classes_direct(self) -> int:
        """P2-B: direct 6-class prediction."""
        return 6

    @property
    def is_multilabel(self) -> bool:
        return self.variation == "D"

    @property
    def is_two_stage(self) -> bool:
        return self.variation in ("C", "D")

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.run_name

    @property
    def results_run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    # ── Serialisation ───────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "P2Config":
        """Load config from YAML."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def __post_init__(self):
        # Auto-generate run_name if still default
        if self.run_name == "p2_D_all_text":
            self.run_name = f"p2_{self.variation}_{self.text_mode}"
