"""Qwen3-VL LoRA Stage-1 config for A6000 48GB."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import yaml

from src.utils.config import PROJECT_ROOT


@dataclass
class S1VLMConfig:
    # Prefer newest Qwen3-VL; fall back handled in trainer if model id fails
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    run_name: str = "s1_qwen3vl_lora"
    text_mode: Literal["tweet", "tweet_ocr", "all_text"] = "all_text"
    ocr_source: str = "filtered"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    epochs: int = 2
    lr: float = 2e-5
    batch_size: int = 2
    grad_accum_steps: int = 16  # effective batch 32
    max_seq_len: int = 1024
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Generative SFT uses hard majority labels (answer tokens). Soft labels
    # would need a regression head; keep False to avoid a misleading flag.
    use_soft_labels: bool = False
    use_4bit: bool = False  # A6000 48GB: full bf16 LoRA preferred
    gradient_checkpointing: bool = True

    num_workers: int = 4
    seed: int = 42
    device: str = "auto"

    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints" / "stage1")
    results_dir: str = str(PROJECT_ROOT / "results" / "stage1")

    def __post_init__(self):
        if self.run_name == "s1_qwen3vl_lora":
            short = self.model_name.split("/")[-1].replace(".", "").lower()
            self.run_name = f"s1_{short}_lora"

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
