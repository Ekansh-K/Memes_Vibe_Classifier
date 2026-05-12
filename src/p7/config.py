"""P7Config — configuration dataclass for all P7 (MHSDF) variations.

Variations:
    A — Binary-only (Stage 1 standalone)
    B — Direct 6-class multi-class (single stage)
    C — Two-stage hierarchical, single-label Stage 2
    D — Two-stage hierarchical, multi-label Stage 2 (primary)

Text modes (caption ablation):
    no_caption — tweet_text + ocr_text        (true baseline)
    tweet_ocr  — tweet_text + ocr_text        (named variant, same input as no_caption)
    all_text   — caption + ocr_text + tweet_text (full context)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.utils.config import PROJECT_ROOT


@dataclass
class P7Config:
    """Full configuration for a single P7 experiment run."""

    # ── Experiment identity ─────────────────────────────────────────────────
    variation: Literal["A", "B", "C", "D"] = "D"
    text_mode: Literal["no_caption", "tweet_ocr", "all_text"] = "tweet_ocr"
    run_name: str = "p7_D_tweet_ocr"

    # ── Data ────────────────────────────────────────────────────────────────
    ocr_source: str = "new"          # "old", "new", or "both"
    img_size: int = 224
    max_seq_len: int = 128
    exclude_full_disagreement: bool = False
    multilabel_threshold: float = 2 / 3   # soft_label[c] >= 2/3 → active

    # ── Tokenizer ───────────────────────────────────────────────────────────
    # cardiffnlp/twitter-roberta-base is optimised for tweet-style text
    bert_model_name: str = "cardiffnlp/twitter-roberta-base"

    # ── CNN visual encoder ──────────────────────────────────────────────────
    cnn_out_dim: int = 512

    # ── BiLSTM text encoder ─────────────────────────────────────────────────
    embed_dim: int = 128              # embedding dimension (learned from BERT vocab)
    lstm_hidden: int = 256            # per-direction hidden size → concat = 512
    lstm_layers: int = 2
    lstm_dropout: float = 0.3

    # ── Classifier head ─────────────────────────────────────────────────────
    head_hidden: int = 256
    head_dropout: float = 0.3

    # ── Stage 1 training ────────────────────────────────────────────────────
    s1_epochs: int = 15
    s1_lr: float = 1e-3
    s1_batch_size: int = 64
    s1_loss: Literal["weighted_bce", "focal"] = "focal"
    s1_focal_gamma: float = 2.0

    # ── Stage 2 training ────────────────────────────────────────────────────
    s2_epochs: int = 20
    s2_lr: float = 5e-4
    s2_batch_size: int = 64
    # Loss is auto-selected: CrossEntropy for C, BCEWithLogits for D

    # ── Optimiser / scheduler ───────────────────────────────────────────────
    weight_decay: float = 5e-4          # raised from 1e-4 — reduces memorisation
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_ratio: float = 0.1
    warmup_epochs: int = 1              # linear warmup epochs (protects GloVe embeddings)
    embed_lr_factor: float = 0.1        # GloVe embedding LR = base_lr × this factor
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1         # 0 = off; 0.1 prevents overconfidence
    early_stop_patience: int = 3         # 0 = disabled; epochs without val improvement

    # ── Hardware ───────────────────────────────────────────────────────────────
    device: str = "auto"             # "auto" → cuda if available, else cpu
    num_workers: int = 4             # 4 for Windows; 8 for Kaggle/Linux
    use_amp: bool = True             # Automatic Mixed Precision (FP16)
    use_data_parallel: bool = True   # nn.DataParallel across all visible CUDA GPUs
    use_image_cache: bool = False    # Legacy RAM dict cache (incompatible w/ workers)
    use_image_store: bool = True     # Memmap image store: eliminates JPEG decode
    use_compile: bool = False        # torch.compile — Linux/Triton only

    # ── Augmentation ───────────────────────────────────────────────────────────
    use_random_erasing: bool = True  # occlude random image patches (anti-overfit)
    token_drop_rate: float = 0.10    # randomly zero tokens during training

    # ── GloVe Twitter embeddings (optional) ──────────────────────────────────
    use_glove: bool = False          # init BiLSTM embeddings from GloVe Twitter
    glove_path: Optional[str] = None # path to glove.twitter.27B.200d.txt
    glove_dim: int = 200             # must match embed_dim when use_glove=True

    # ── Dataset size control ──────────────────────────────────────────────────
    # None = use full split; int = stratified subset (preserves class ratio)
    max_train_samples: Optional[int] = None
    max_val_samples:   Optional[int] = None

    # ── Paths ───────────────────────────────────────────────────────────────
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "p7")
    results_dir: str = str(PROJECT_ROOT / "results" / "p7")

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    log_every_n_steps: int = 50

    # ── Derived properties ───────────────────────────────────────────────────
    @property
    def num_classes_s1(self) -> int:
        return 1

    @property
    def num_classes_s2(self) -> int:
        """Stage 2 predicts over 5 hate categories (excludes NotHate=0)."""
        return 5

    @property
    def num_classes_direct(self) -> int:
        """P7-B: direct 6-class prediction."""
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

    # ── Serialisation ────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        """Save config to YAML."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "P7Config":
        """Load config from YAML."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def __post_init__(self):
        # Auto-generate run_name if still default
        if self.run_name == "p7_D_tweet_ocr":
            self.run_name = f"p7_{self.variation}_{self.text_mode}"
