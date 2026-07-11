"""Config for Stage-1 text full fine-tune (A6000 48GB defaults)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.utils.config import PROJECT_ROOT

MODEL_PRESETS = {
    "twitter-roberta": "cardiffnlp/twitter-roberta-base",
    "hate-latest": "cardiffnlp/twitter-roberta-base-hate-latest",
    "hatebert": "GroNLP/hateBERT",
    "deberta-large": "microsoft/deberta-v3-large",
    "roberta-large": "roberta-large",
}


@dataclass
class S1TextConfig:
    model_key: str = "hate-latest"
    model_name: str = ""  # filled from preset if empty
    text_mode: Literal["tweet", "tweet_ocr", "all_text"] = "all_text"
    ocr_source: str = "filtered"
    run_name: str = "s1_text_hate_latest"

    # Training — A6000 friendly
    epochs: int = 4
    lr: float = 2e-5
    batch_size: int = 64
    grad_accum_steps: int = 1
    max_seq_len: int = 192
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    early_stop_patience: int = 2
    label_smoothing: float = 0.0

    use_soft_labels: bool = True
    use_agreement_weighting: bool = True
    agreement_weights: tuple = (0.4, 0.7, 1.0)

    use_amp: bool = True
    amp_dtype: str = "bf16"  # bf16 on A6000 Ampere+
    num_workers: int = 8
    seed: int = 42
    device: str = "auto"

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    exclude_full_disagreement: bool = False

    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "stage1")
    results_dir: str = str(PROJECT_ROOT / "results" / "stage1")

    def __post_init__(self):
        if not self.model_name:
            if self.model_key not in MODEL_PRESETS:
                raise ValueError(
                    f"Unknown model_key={self.model_key}. "
                    f"Choose from {list(MODEL_PRESETS)} or set model_name."
                )
            self.model_name = MODEL_PRESETS[self.model_key]
        if self.run_name == "s1_text_hate_latest":
            self.run_name = f"s1_text_{self.model_key}_{self.text_mode}"
        # DeBERTa-large: smaller default batch
        if "large" in self.model_name.lower() and self.batch_size > 48:
            self.batch_size = 32

    @property
    def run_dir(self) -> Path:
        return Path(self.checkpoint_dir) / self.run_name

    @property
    def results_run_dir(self) -> Path:
        return Path(self.results_dir) / self.run_name

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
