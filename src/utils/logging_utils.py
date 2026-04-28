"""Weights & Biases logging helpers and reproducibility utilities."""

import random
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_wandb(
    config: dict,
    project_name: str = "mmhs150k-hate-speech",
    run_name: Optional[str] = None,
    enabled: bool = True,
) -> Optional[object]:
    """Initialize a W&B run. Returns the run object or None if disabled."""
    if not enabled:
        return None
    try:
        import wandb

        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True,
        )
        return run
    except ImportError:
        print("[WARN] wandb not installed — experiment tracking disabled.")
        return None
    except Exception as e:
        print(f"[WARN] wandb init failed: {e} — running without tracking.")
        return None


def log_metrics(
    metrics: dict,
    step: Optional[int] = None,
) -> None:
    """Log metrics to W&B if a run is active."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except ImportError:
        pass


def log_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list,
    title: str = "Confusion Matrix",
) -> None:
    """Log a confusion matrix to W&B as a wandb.Table."""
    try:
        import wandb
        from sklearn.metrics import confusion_matrix

        if wandb.run is None:
            return
        cm = confusion_matrix(y_true, y_pred)
        data = []
        for i, row_name in enumerate(class_names):
            for j, col_name in enumerate(class_names):
                data.append([row_name, col_name, int(cm[i][j])])
        table = wandb.Table(data=data, columns=["Actual", "Predicted", "Count"])
        wandb.log({title: table})
    except ImportError:
        pass
