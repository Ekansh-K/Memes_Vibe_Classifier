"""Multi-label metrics for P7-D evaluation.

Extends src/evaluation/metrics.py with:
- compute_multilabel_metrics: Micro/macro/sample F1, Hamming loss, exact match, Jaccard
- calibrate_thresholds:       Per-category threshold search on validation logits
- compute_pipeline_metrics:   End-to-end two-stage pipeline metrics

All functions are pure (no side effects) and return dicts compatible
with the existing metrics logging format.
"""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# Hate category names for Stage 2 (5 categories, NotHate excluded)
HATE_CAT_NAMES = ["Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    category_names: Optional[list[str]] = None,
) -> dict:
    """Compute comprehensive multi-label classification metrics.

    Args:
        y_true:         (N, C) binary ground-truth matrix.
        y_pred:         (N, C) binary predicted matrix.
        category_names: List of C category names for per-class keys.

    Returns:
        Dict with micro/macro/sample F1, Hamming loss, exact match, Jaccard,
        per-class F1, precision, recall.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    names = category_names or HATE_CAT_NAMES

    metrics = {
        "multilabel/micro_f1":    float(f1_score(y_true, y_pred, average="micro",    zero_division=0)),
        "multilabel/macro_f1":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "multilabel/sample_f1":   float(f1_score(y_true, y_pred, average="samples",  zero_division=0)),
        "multilabel/hamming_loss":float(hamming_loss(y_true, y_pred)),
        "multilabel/exact_match": float(accuracy_score(y_true, y_pred)),
    }

    try:
        metrics["multilabel/jaccard"] = float(
            jaccard_score(y_true, y_pred, average="samples", zero_division=0)
        )
    except Exception:
        metrics["multilabel/jaccard"] = 0.0

    # Per-class metrics
    per_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)

    for i, name in enumerate(names):
        if i < len(per_f1):
            metrics[f"multilabel/{name}/f1"]        = float(per_f1[i])
            metrics[f"multilabel/{name}/precision"]  = float(per_prec[i])
            metrics[f"multilabel/{name}/recall"]     = float(per_rec[i])

    return metrics


def calibrate_thresholds(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    threshold_range: tuple[float, float, float] = (0.10, 0.90, 0.05),
    metric: str = "macro_f1",
) -> np.ndarray:
    """Per-category threshold calibration via grid search on validation set.

    Searches independently for each category; chooses threshold that maximises
    the specified metric on the validation set.

    Args:
        val_logits:       (N, C) raw logits from Stage 2 model on val set.
        val_labels:       (N, C) binary ground-truth multi-label matrix.
        threshold_range:  (start, stop, step) for np.arange.
        metric:           "macro_f1" (default) or "micro_f1".

    Returns:
        (C,) array of optimal per-category thresholds.
    """
    val_probs = 1 / (1 + np.exp(-val_logits))   # sigmoid
    n_cats = val_labels.shape[1]
    thresholds = np.full(n_cats, 0.5)

    start, stop, step = threshold_range
    grid = np.arange(start, stop + step / 2, step)

    for c in range(n_cats):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            preds_c = (val_probs[:, c] >= t).astype(int)
            f1 = float(f1_score(val_labels[:, c], preds_c, zero_division=0))
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
        cat_name = HATE_CAT_NAMES[c] if c < len(HATE_CAT_NAMES) else str(c)
        logger.info(
            f"[P7 Calibration] {cat_name}: best_threshold={best_t:.2f}  F1={best_f1:.4f}"
        )

    return thresholds


def apply_thresholds(
    logits: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Convert logits to binary predictions using per-category thresholds.

    Args:
        logits:     (N, C) raw logits.
        thresholds: (C,) per-category thresholds.

    Returns:
        (N, C) binary prediction matrix.
    """
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= thresholds).astype(int)


def compute_pipeline_metrics(
    s1_true: np.ndarray,
    s1_pred: np.ndarray,
    s2_true,
    s2_pred,
    multilabel: bool = False,
) -> dict:
    """Compute end-to-end two-stage pipeline metrics.

    Stage 1 recall is critical — missing a hateful sample is a hard error.
    The composite score = S1_recall_hate * S2_macro_f1.

    Args:
        s1_true:    (N,) binary ground truth (0=NotHate, 1=Hate).
        s1_pred:    (N,) binary predictions.
        s2_true:    Stage 2 ground truth (single int or multi-hot array).
        s2_pred:    Stage 2 predictions.
        multilabel: If True, compute multilabel metrics for Stage 2.

    Returns:
        Dict with composite and stage-level metrics.
    """
    s1_true = np.asarray(s1_true)
    s1_pred = np.asarray(s1_pred)

    s1_recall_hate = float(
        recall_score(s1_true, s1_pred, pos_label=1, zero_division=0)
    )
    s1_precision_hate = float(
        precision_score(s1_true, s1_pred, pos_label=1, zero_division=0)
    )
    s1_macro_f1 = float(f1_score(s1_true, s1_pred, average="macro", zero_division=0))

    metrics = {
        "pipeline/s1_recall_hate":    s1_recall_hate,
        "pipeline/s1_precision_hate": s1_precision_hate,
        "pipeline/s1_macro_f1":       s1_macro_f1,
    }

    if s2_true is not None and s2_pred is not None:
        s2_true = np.asarray(s2_true)
        s2_pred = np.asarray(s2_pred)

        if multilabel:
            s2_macro_f1 = float(
                f1_score(s2_true, s2_pred, average="macro", zero_division=0)
            )
        else:
            s2_macro_f1 = float(
                f1_score(s2_true, s2_pred, average="macro", zero_division=0)
            )

        metrics["pipeline/s2_macro_f1"] = s2_macro_f1
        metrics["pipeline/composite"] = s1_recall_hate * s2_macro_f1

    return metrics
