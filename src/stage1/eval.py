"""Stage-1 evaluation: metrics, threshold sweep, result IO."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_stage1_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Full Stage-1 metric dict at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "hate_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "hate_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "hate_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "nothate_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "n_samples": int(len(y_true)),
        "n_hate": int(y_true.sum()),
    }
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.0
    return metrics


def sweep_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    min_hate_precision: float = 0.0,
) -> dict:
    """Sweep thresholds; return best-by-macro-F1 and best-by-hate-recall (with optional precision floor)."""
    if thresholds is None:
        thresholds = np.linspace(0.10, 0.90, 81)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    best_f1 = {"macro_f1": -1.0}
    best_recall = {"hate_recall": -1.0}

    for t in thresholds:
        m = compute_stage1_metrics(y_true, y_prob, threshold=float(t))
        if m["macro_f1"] > best_f1["macro_f1"]:
            best_f1 = m
        if m["hate_precision"] + 1e-9 >= min_hate_precision:
            if m["hate_recall"] > best_recall["hate_recall"] or (
                m["hate_recall"] == best_recall.get("hate_recall", -1)
                and m["macro_f1"] > best_recall.get("macro_f1", -1)
            ):
                best_recall = m

    if best_recall.get("hate_recall", -1) < 0:
        best_recall = best_f1

    return {
        "best_macro_f1": best_f1,
        "best_hate_recall": best_recall,
        "at_0.5": compute_stage1_metrics(y_true, y_prob, threshold=0.5),
    }


def brier_score(y_true_or_soft: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between targets (hard or soft) and predicted probs."""
    yt = np.asarray(y_true_or_soft, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    return float(np.mean((yp - yt) ** 2))


def agreement_stratified_stage1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    agreement: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Macro F1 / AUC per binary agreement level (1/2/3)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    agreement = np.asarray(agreement).astype(int)
    out = {}
    for level, name in ((3, "unanimous"), (2, "majority"), (1, "split")):
        mask = agreement == level
        n = int(mask.sum())
        key = f"agreement_{name}"
        if n < 5:
            out[key] = {"n": n, "macro_f1": None, "auc_roc": None}
            continue
        m = compute_stage1_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        out[key] = {
            "n": n,
            "macro_f1": m["macro_f1"],
            "hate_f1": m["hate_f1"],
            "auc_roc": m["auc_roc"],
            "hate_recall": m["hate_recall"],
        }
    return out


def evaluate_stage1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_hate_precision: float = 0.40,
    agreement: Optional[np.ndarray] = None,
    soft_targets: Optional[np.ndarray] = None,
) -> dict:
    """Convenience: full Stage-1 report with threshold sweep.

    Optional:
      agreement — per-sample binary agreement levels for stratified metrics
      soft_targets — soft P(hate) for Brier vs soft distribution
    """
    sweep = sweep_threshold(y_true, y_prob, min_hate_precision=min_hate_precision)
    thr = sweep["best_macro_f1"]["threshold"]
    out = {
        "at_best_macro_f1": sweep["best_macro_f1"],
        "at_best_hate_recall": sweep["best_hate_recall"],
        "at_0.5": sweep["at_0.5"],
        # Flatten primary numbers for quick logging
        "macro_f1": sweep["best_macro_f1"]["macro_f1"],
        "hate_recall": sweep["best_macro_f1"]["hate_recall"],
        "hate_f1": sweep["best_macro_f1"]["hate_f1"],
        "auc_roc": sweep["best_macro_f1"]["auc_roc"],
        "threshold": thr,
    }
    # Calibration / soft fidelity
    out["brier_hard"] = brier_score(y_true, y_prob)
    if soft_targets is not None:
        out["brier_soft"] = brier_score(soft_targets, y_prob)
    if agreement is not None:
        out["stratified"] = agreement_stratified_stage1(
            y_true, y_prob, agreement, threshold=thr
        )
    logger.info(
        f"[S1 Eval] macro_f1={out['macro_f1']:.4f}  hate_f1={out['hate_f1']:.4f}  "
        f"hate_recall={out['hate_recall']:.4f}  auc={out['auc_roc']:.4f}  "
        f"thr={out['threshold']:.2f}  brier_hard={out['brier_hard']:.4f}"
    )
    return out


def save_stage1_results(
    run_dir: Path,
    metrics: dict,
    y_true: Optional[np.ndarray] = None,
    y_prob: Optional[np.ndarray] = None,
    tweet_ids: Optional[list] = None,
) -> None:
    """Write metrics.json and optional val_preds.npz for ensemble."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if y_true is not None and y_prob is not None:
        payload = {"y_true": np.asarray(y_true), "y_prob": np.asarray(y_prob)}
        if tweet_ids is not None:
            payload["tweet_ids"] = np.asarray(tweet_ids)
        np.savez_compressed(run_dir / "val_preds.npz", **payload)
    logger.info(f"[S1 Eval] Saved results → {run_dir}")
