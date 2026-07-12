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
    # Soft-label recipe S0–S7 (overrides use_soft_labels / agreement when set)
    soft_recipe: str = "S2"
    use_binary_agreement: bool = True
    multi_task_lambda: float = 0.5
    soft_temperature: float = 1.0
    min_agreement_binary: Optional[int] = None
    early_stop_metric: str = "macro_f1"  # or "auc_roc"

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
        # Apply soft recipe defaults (may set soft/agreement flags)
        if self.soft_recipe:
            from src.stage1.soft_recipes import get_soft_recipe

            spec = get_soft_recipe(self.soft_recipe)
            self.use_soft_labels = spec.use_soft_labels if not spec.multi_task else True
            self.use_agreement_weighting = spec.use_agreement_weighting
            self.agreement_weights = tuple(spec.agreement_weights)
            self.use_binary_agreement = spec.use_binary_agreement
            self.multi_task_lambda = spec.multi_task_lambda
            self.soft_temperature = spec.soft_temperature
            if spec.min_agreement_binary is not None:
                self.min_agreement_binary = spec.min_agreement_binary
        if self.run_name == "s1_text_hate_latest":
            recipe_tag = (self.soft_recipe or "S2").upper()
            self.run_name = f"s1_text_{self.model_key}_{self.text_mode}_{recipe_tag}"
        # Large models: smaller batch + safer AMP (bf16 often NaNs on DeBERTa-v3)
        if "large" in self.model_name.lower():
            if self.batch_size > 32:
                self.batch_size = 16
            if self.lr > 1e-5:
                self.lr = 1e-5
            if self.amp_dtype == "bf16":
                self.amp_dtype = "fp16"

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
