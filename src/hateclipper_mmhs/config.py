"""Hate-CLIPper MMHS config — A6000 defaults; align fusion preferred."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.utils.config import PROJECT_ROOT


@dataclass
class HateCLIPperConfig:
    run_name: str = "s1_hateclipper_align"
    text_mode: Literal["tweet", "tweet_ocr", "all_text"] = "all_text"
    ocr_source: str = "filtered"

    # CLIP via HuggingFace (reliable install)
    clip_model: str = "openai/clip-vit-large-patch14"
    map_dim: int = 512
    fusion: Literal["align", "concat", "cross"] = "align"
    num_mapping_layers: int = 1
    num_pre_output_layers: int = 1
    drop_map: float = 0.1
    drop_fusion: float = 0.4
    drop_pre: float = 0.2
    freeze_encoders: bool = True

    # MemeCLIP-style adapters (optional)
    use_adapters: bool = True
    adapter_ratio: float = 0.2  # residual mix α

    # Training
    epochs: int = 12
    lr: float = 1e-4
    batch_size: int = 64
    grad_accum_steps: int = 1
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    early_stop_patience: int = 3

    use_soft_labels: bool = True
    use_agreement_weighting: bool = True
    agreement_weights: tuple = (0.4, 0.7, 1.0)
    soft_recipe: str = "S2"
    use_binary_agreement: bool = True
    multi_task_lambda: float = 0.5
    soft_temperature: float = 1.0
    min_agreement_binary: Optional[int] = None
    early_stop_metric: str = "macro_f1"
    # Partial unfreeze: unfreeze last N transformer blocks of CLIP vision+text
    unfreeze_last_n: int = 0
    backbone_lr: float = 1e-6

    use_amp: bool = True
    amp_dtype: str = "bf16"
    num_workers: int = 8
    seed: int = 42
    device: str = "auto"
    img_size: int = 224
    use_image_store: bool = True

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "stage1")
    results_dir: str = str(PROJECT_ROOT / "results" / "stage1")

    def __post_init__(self):
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
        if self.run_name == "s1_hateclipper_align":
            self.run_name = f"s1_hateclipper_{self.fusion}"
            if self.use_adapters:
                self.run_name += "_adapters"
            recipe_tag = (self.soft_recipe or "S2").upper()
            self.run_name += f"_{recipe_tag}"
            if self.unfreeze_last_n > 0:
                self.run_name += f"_uf{self.unfreeze_last_n}"
        if self.fusion == "cross" and self.map_dim > 256:
            # n² features; cap for VRAM/params
            self.map_dim = 256

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
