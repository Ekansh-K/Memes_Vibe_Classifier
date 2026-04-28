"""Evaluation metrics for MMHS150K hate speech detection.

Primary metric: Macro F1 (handles class imbalance).
All functions return dicts directly loggable to W&B.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.config import LABEL_MAP_FINE, LABEL_MAP_BINARY


def compute_binary_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: Optional[list | np.ndarray] = None,
) -> dict:
    """Compute all binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_prob: Predicted probabilities for the positive class (for AUC-ROC).

    Returns:
        Dict with accuracy, macro_f1, weighted_f1, precision, recall, auc_roc.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "binary/accuracy": float(accuracy_score(y_true, y_pred)),
        "binary/macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "binary/weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "binary/precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "binary/recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["binary/auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["binary/auc_roc"] = 0.0  # single class in batch

    return metrics


def compute_multiclass_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> dict:
    """Compute all multiclass classification metrics.

    Args:
        y_true: Ground truth labels (0-5).
        y_pred: Predicted labels (0-5).
        y_prob: Predicted probability matrix (N, 6) for AUC-ROC.
        class_names: Names for each class. Defaults to LABEL_MAP_FINE values.

    Returns:
        Dict with overall and per-class metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_names is None:
        class_names = list(LABEL_MAP_FINE.values())

    metrics = {
        "multiclass/accuracy": float(accuracy_score(y_true, y_pred)),
        "multiclass/macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "multiclass/weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "multiclass/precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "multiclass/recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, name in enumerate(class_names):
        if i < len(per_class_f1):
            metrics[f"multiclass/{name}/f1"] = float(per_class_f1[i])
            metrics[f"multiclass/{name}/precision"] = float(per_class_precision[i])
            metrics[f"multiclass/{name}/recall"] = float(per_class_recall[i])

    # AUC-ROC (one-vs-rest)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["multiclass/auc_roc_ovr"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except ValueError:
            metrics["multiclass/auc_roc_ovr"] = 0.0

    # Confusion matrix as nested list
    cm = confusion_matrix(y_true, y_pred)
    metrics["multiclass/confusion_matrix"] = cm.tolist()

    return metrics


def compute_agreement_stratified_metrics(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    agreement_levels: list | np.ndarray,
) -> dict:
    """Compute metrics stratified by annotator agreement level.

    Useful for understanding model performance on easy (unanimous) vs
    hard (disagreed) samples.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        agreement_levels: Per-sample agreement level (3=unanimous, 2=majority, 1=all-differ).

    Returns:
        Dict with metrics per agreement level.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    agreement_levels = np.asarray(agreement_levels)

    metrics = {}
    for level in [3, 2, 1]:
        mask = agreement_levels == level
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        level_name = {3: "unanimous", 2: "majority", 1: "all_disagree"}[level]
        metrics[f"agreement_{level_name}/count"] = int(mask.sum())
        metrics[f"agreement_{level_name}/accuracy"] = float(accuracy_score(yt, yp))
        metrics[f"agreement_{level_name}/macro_f1"] = float(
            f1_score(yt, yp, average="macro", zero_division=0)
        )

    return metrics


def format_classification_report(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    class_names: Optional[list[str]] = None,
) -> str:
    """Return a formatted classification report string."""
    if class_names is None:
        class_names = list(LABEL_MAP_FINE.values())
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
