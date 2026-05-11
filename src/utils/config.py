"""Experiment configuration using dataclasses. Serializable to/from YAML."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import yaml


# ── Project paths (always relative to PROJECT_ROOT) ──────────────────────────
import os as _os

PROJECT_ROOT = Path(_os.environ.get("MMHS_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])))

# Data directory: can be overridden to point at a read-only Kaggle input path
# while PROJECT_ROOT (checkpoints, results, caches) stays on writable storage.
_DATA_DIR_ENV = _os.environ.get("MMHS_DATA_DIR")
DATASET_DIR = Path(_DATA_DIR_ENV) if _DATA_DIR_ENV else PROJECT_ROOT / "dataset"

GT_FILE              = DATASET_DIR / "MMHS150K_GT.json"
IMG_DIR              = DATASET_DIR / "img_resized"
OCR_DIR_OLD          = DATASET_DIR / "img_txt"
OCR_DIR_NEW          = DATASET_DIR / "img_txt_new"
OCR_CONSOLIDATED     = DATASET_DIR / "ocr_consolidated.json"
PROCESSED_LABELS_FILE = DATASET_DIR / "processed_labels.json"
SPLITS_DIR           = DATASET_DIR / "splits"
CHECKPOINTS_DIR      = PROJECT_ROOT / "checkpoints"
RESULTS_DIR          = PROJECT_ROOT / "results"

# ── Dataset constants ────────────────────────────────────────────────────────
NUM_CLASSES_BINARY = 2
NUM_CLASSES_FINE = 6
LABEL_MAP_FINE = {
    0: "NotHate",
    1: "Racist",
    2: "Sexist",
    3: "Homophobe",
    4: "Religion",
    5: "OtherHate",
}
LABEL_MAP_BINARY = {0: "NotHate", 1: "Hate"}
TOTAL_SAMPLES = 149_823
TRAIN_SIZE = 134_823
VAL_SIZE = 5_000
TEST_SIZE = 10_000


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_dir: str = str(DATASET_DIR)
    img_dir: str = str(IMG_DIR)
    ocr_source: str = "new"  # "old", "new", or "both"
    img_size: int = 224
    convert_emoji: bool = False  # whether to convert emoji to text description
    exclude_full_disagreement: bool = False  # drop samples where 3 annotators all disagree
    max_tweet_length: int = 512
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Configuration for training."""

    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    loss_type: str = "focal"  # "focal", "ce_weighted", "soft_ce"
    focal_gamma: float = 2.0
    contrastive_weight: float = 0.1
    label_smoothing: float = 0.0
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    mixed_precision: str = "bf16"  # "bf16", "fp16", "fp32"
    seed: int = 42


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    text_encoder: str = "roberta-base"
    image_encoder: str = "openai/clip-vit-base-patch32"
    fusion_type: str = "cross_attention"  # "cross_attention", "concat", "late_fusion"
    hidden_dim: int = 512
    num_attention_heads: int = 8
    num_fusion_layers: int = 2
    dropout: float = 0.1
    freeze_text_encoder: bool = False
    freeze_image_encoder: bool = True
    num_classes: int = NUM_CLASSES_FINE
    classification_mode: str = "multiclass"  # "binary" or "multiclass"


@dataclass
class ExperimentConfig:
    """Top-level configuration combining all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run_name: str = "default_run"
    track: str = "A"  # "A", "B", or "C"
    wandb_project: str = "mmhs150k-hate-speech"
    wandb_enabled: bool = True
    notes: str = ""

    def save(self, path: Path) -> None:
        """Save config to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            train=TrainConfig(**raw.get("train", {})),
            model=ModelConfig(**raw.get("model", {})),
            run_name=raw.get("run_name", "default_run"),
            track=raw.get("track", "A"),
            wandb_project=raw.get("wandb_project", "mmhs150k-hate-speech"),
            wandb_enabled=raw.get("wandb_enabled", True),
            notes=raw.get("notes", ""),
        )
