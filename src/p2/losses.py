"""Loss function factory for all P2 variations and stages.

Loss selection:
    P2-A          → BCEWithLogitsLoss (weighted, binary)
    P2-B          → CrossEntropyLoss  (weighted, 6-class)
    P2-C Stage 1  → BCEWithLogitsLoss (weighted, binary)
    P2-C Stage 2  → CrossEntropyLoss  (weighted, 5-class, hateful only)
    P2-D Stage 1  → BCEWithLogitsLoss (weighted, binary)
    P2-D Stage 2  → BCEWithLogitsLoss (per-category weighted, 5-way multilabel)

All weights are computed using inverse-frequency weighting:
    weight[c] = total / (num_classes * count[c])
"""

import logging
from collections import Counter

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_binary_pos_weight(labels: list[int], device: torch.device) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss (binary).

    pos_weight = n_negative / n_positive
    """
    counts = Counter(labels)
    n_pos = counts.get(1, 1)
    n_neg = counts.get(0, 1)
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    logger.info(f"[P2 Loss] Binary pos_weight: {pw.item():.3f}  (neg={n_neg}, pos={n_pos})")
    return pw


def compute_class_weights(
    labels: list[int], num_classes: int, device: torch.device,
    max_weight: float = 100.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights, capped at max_weight.

    weight[c] = total / (num_classes * count[c])

    Capped at max_weight (default=50) to avoid training instability for
    extremely rare classes (e.g. Religion at 0.1% gives weight=171 uncapped).
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        w = min(total / (num_classes * cnt), max_weight)
        weights.append(w)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"[P2 Loss] Class weights ({num_classes}-class, capped@{max_weight}): {w.tolist()}")
    return w


def compute_multilabel_pos_weights(
    multilabel_vectors: list[list[float]],
    num_categories: int,
    device: torch.device,
    max_pos_weight: float = 100.0,
) -> torch.Tensor:
    """Compute per-category pos_weight for multilabel BCEWithLogitsLoss.

    For each category c:
        pos_weight[c] = n_negative[c] / n_positive[c]

    Capped at max_pos_weight (default=100) to prevent gradient instability
    for extremely rare classes (e.g. Religion at ~0.6% of Stage 2 data
    would give pos_weight ~156 without capping; cap=100 gives 66% of proper signal
    vs cap=50 which gave only 33%).
    """
    arr = torch.tensor(multilabel_vectors, dtype=torch.float32)
    n_pos = arr.sum(dim=0).clamp(min=1)
    n_neg = len(multilabel_vectors) - n_pos
    pw_raw = n_neg / n_pos

    # Cap extreme weights and warn
    pw_capped = pw_raw.clamp(max=max_pos_weight)
    capped_mask = pw_raw > max_pos_weight
    if capped_mask.any():
        logger.warning(
            f"[P2 Loss] Multilabel pos_weights capped at {max_pos_weight} for "
            f"categories at indices {capped_mask.nonzero(as_tuple=True)[0].tolist()}. "
            f"Raw values: {pw_raw[capped_mask].tolist()} — likely very rare classes. "
            f"Consider oversampling or focal loss if these classes underperform."
        )

    pw = pw_capped.to(device)
    logger.info(
        f"[P2 Loss] Multilabel pos_weights (capped@{max_pos_weight}): {pw.tolist()}"
    )
    return pw


def get_p2_loss(
    variation: str,
    stage: int,
    train_dataset,
    device: torch.device,
    label_smoothing: float = 0.0,
):
    """Return the appropriate loss function for the given P2 variant and stage.

    Args:
        variation:      "A", "B", "C", or "D".
        stage:          1 or 2 (for C/D); ignored for A/B (use stage=1).
        train_dataset:  P2Dataset instance for the training split.
        device:         Target device.
        label_smoothing: Amount of label smoothing.

    Returns:
        Instantiated loss nn.Module.
    """
    labels_binary = [
        train_dataset.labels[sid]["hard_label_binary"]
        for sid in train_dataset.sample_ids
    ]
    labels_6class = [
        train_dataset.labels[sid]["hard_label_6class"]
        for sid in train_dataset.sample_ids
    ]

    # ── P2-A or Stage 1 of C/D ────────────────────────────────────────────
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        pw = compute_binary_pos_weight(labels_binary, device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── P2-B — direct 6-class ─────────────────────────────────────────────
    if variation == "B":
        weights = compute_class_weights(labels_6class, num_classes=6, device=device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    # ── P2-C Stage 2 — 5-class single-label ─────────────────────────────
    if variation == "C" and stage == 2:
        labels_s2 = [
            train_dataset.labels[sid]["hard_label_6class"] - 1
            for sid in train_dataset.sample_ids
            if train_dataset.labels[sid]["hard_label_6class"] > 0
        ]
        weights = compute_class_weights(labels_s2, num_classes=5, device=device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    # ── P2-D Stage 2 — 5-way multilabel ─────────────────────────────────
    if variation == "D" and stage == 2:
        threshold = train_dataset.config.multilabel_threshold
        ml_vecs = []
        for sid in train_dataset.sample_ids:
            soft = train_dataset.labels[sid]["soft_label_6class"]
            vec = [1.0 if soft[c] >= threshold else 0.0 for c in range(1, 6)]
            ml_vecs.append(vec)
        pw = compute_multilabel_pos_weights(ml_vecs, num_categories=5, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    raise ValueError(f"Unknown combination: variation={variation!r}, stage={stage}")
