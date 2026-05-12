"""Loss function factory for all P7 variations and stages.

Loss selection:
    P7-A          → BCEWithLogitsLoss (weighted, binary)
    P7-B          → CrossEntropyLoss  (weighted, 6-class)
    P7-C Stage 1  → BCEWithLogitsLoss (weighted, binary)
    P7-C Stage 2  → CrossEntropyLoss  (weighted, 5-class, hateful only)
    P7-D Stage 1  → BCEWithLogitsLoss (weighted, binary)
    P7-D Stage 2  → BCEWithLogitsLoss (per-category weighted, 5-way multilabel)

All weights are computed using inverse-frequency weighting:
    weight[c] = total / (num_classes * count[c])
"""

import logging
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Weight computation helpers ────────────────────────────────────────────────

def compute_class_weights(
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    weight[c] = total / (num_classes * count[c])

    Args:
        labels:      List of integer class labels in the training split.
        num_classes: Total number of classes.
        device:      Target device for the weight tensor.

    Returns:
        FloatTensor (num_classes,) of class weights.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)   # avoid divide-by-zero with floor of 1
        weights.append(total / (num_classes * cnt))
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"[P7 Loss] Class weights ({num_classes}-class): {w.tolist()}")
    return w


def compute_binary_pos_weight(
    labels: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss (binary).

    pos_weight = n_negative / n_positive

    Args:
        labels: List of binary labels (0=NotHate, 1=Hate).
        device: Target device.

    Returns:
        Scalar FloatTensor.
    """
    counts = Counter(labels)
    n_pos = counts.get(1, 1)
    n_neg = counts.get(0, 1)
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    logger.info(f"[P7 Loss] Binary pos_weight: {pw.item():.3f}  (neg={n_neg}, pos={n_pos})")
    return pw


def compute_multilabel_pos_weights(
    multilabel_vectors: list[list[float]],
    num_categories: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-category pos_weight for multilabel BCEWithLogitsLoss.

    For each category c:
        pos_weight[c] = n_negative[c] / n_positive[c]

    Args:
        multilabel_vectors: List of N multi-hot vectors (each length num_categories).
        num_categories:     Number of hate categories (5 for Stage 2).
        device:             Target device.

    Returns:
        FloatTensor (num_categories,) of per-category pos_weights.
    """
    import numpy as np
    arr = torch.tensor(multilabel_vectors, dtype=torch.float32)
    n_pos = arr.sum(dim=0).clamp(min=1)          # (num_categories,)
    n_neg = len(multilabel_vectors) - n_pos
    pw = (n_neg / n_pos).to(device)
    logger.info(f"[P7 Loss] Multilabel pos_weights: {pw.tolist()}")
    return pw


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss for Stage 1 imbalanced binary classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma:      Focusing parameter (default 2.0).
        pos_weight: Optional scalar pos_weight for positive class.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B,) or (B, 1) raw logits.
            targets: (B,) binary float targets (0.0 or 1.0).
        """
        logits = logits.view(-1)
        targets = targets.float().view(-1)
        # Apply label smoothing to focal loss targets too
        if hasattr(self, 'smoothing') and self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
        )
        p_t = torch.exp(-bce)
        focal = ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


# ── Label-smoothed BCE ───────────────────────────────────────────────────────

class SmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing for binary targets.

    Converts hard 0/1 targets to soft targets:
        1 → 1 - smoothing/2
        0 → smoothing/2

    Prevents the model from driving logits to ±∞ to achieve zero loss,
    which is the root cause of train_loss collapsing to 0.12 while
    validation performance degrades.
    """

    def __init__(self, pos_weight: torch.Tensor, smoothing: float = 0.1):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.smoothing > 0:
            targets = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, targets.float(),
            pos_weight=self.pos_weight,
        )


# ── Main factory ──────────────────────────────────────────────────────────────

def get_p7_loss(
    variation: str,
    stage: int,
    train_dataset,
    device: torch.device,
    focal_gamma: float = 2.0,
    s1_loss_type: str = "focal",
    label_smoothing: float = 0.0,
):
    """Return the appropriate loss function for the given P7 variant and stage.

    Args:
        variation:      "A", "B", "C", or "D".
        stage:          1 or 2 (for C/D); ignored for A/B (use stage=1).
        train_dataset:  P7Dataset instance for the training split (used to
                        compute class statistics for weighting).
        device:         Target device.
        focal_gamma:    Gamma for FocalLoss (only used when s1_loss_type="focal").
        s1_loss_type:   "weighted_bce" or "focal" for Stage 1 / P7-A.
        label_smoothing: Amount of label smoothing to apply.

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

    # ── P7-A or Stage 1 of C/D ────────────────────────────────────────────
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        if s1_loss_type == "focal":
            counts = Counter(labels_binary)
            pw = counts.get(0, 1) / max(counts.get(1, 1), 1)
            logger.info(f"[P7 Loss] Using FocalLoss (gamma={focal_gamma}, pos_weight={pw:.3f}, smoothing={label_smoothing})")
            return FocalLoss(gamma=focal_gamma, pos_weight=pw, smoothing=label_smoothing)
        else:
            pw = compute_binary_pos_weight(labels_binary, device)
            if label_smoothing > 0:
                logger.info(f"[P7 Loss] BCE + label_smoothing={label_smoothing}")
                return SmoothedBCEWithLogitsLoss(pw, smoothing=label_smoothing)
            return nn.BCEWithLogitsLoss(pos_weight=pw)

    # ── P7-B — direct 6-class ─────────────────────────────────────────────
    if variation == "B":
        weights = compute_class_weights(labels_6class, num_classes=6, device=device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    # ── P7-C Stage 2 — 5-class single-label ─────────────────────────────
    if variation == "C" and stage == 2:
        # Stage 2 dataset already filtered to hateful-only;
        # labels are 0-4 (remapped via HATE_CATEGORIES)
        labels_s2 = [
            train_dataset.labels[sid]["hard_label_6class"] - 1
            for sid in train_dataset.sample_ids
            if train_dataset.labels[sid]["hard_label_6class"] > 0
        ]
        weights = compute_class_weights(labels_s2, num_classes=5, device=device)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    # ── P7-D Stage 2 — 5-way multilabel ─────────────────────────────────
    if variation == "D" and stage == 2:
        # Build multilabel vectors for all training samples in stage-2 dataset
        # (hateful-only subset, 5 categories = indices 1-5 of soft_label)
        ml_vecs = []
        threshold = train_dataset.config.multilabel_threshold
        for sid in train_dataset.sample_ids:
            soft = train_dataset.labels[sid]["soft_label_6class"]
            # Extract hate categories only (indices 1-5)
            vec = [1.0 if soft[c] >= threshold else 0.0 for c in range(1, 6)]
            ml_vecs.append(vec)
        pw = compute_multilabel_pos_weights(ml_vecs, num_categories=5, device=device)
        if label_smoothing > 0:
            # Wrap with a custom module that applies smoothing before BCE
            class _SmoothedMultilabelBCE(nn.Module):
                def __init__(self, pos_weight, smoothing):
                    super().__init__()
                    self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    self.smoothing = smoothing
                def forward(self, logits, targets):
                    targets = targets.float() * (1.0 - self.smoothing) + 0.5 * self.smoothing
                    return self.bce(logits, targets)
            return _SmoothedMultilabelBCE(pw, label_smoothing)
        return nn.BCEWithLogitsLoss(pos_weight=pw)

    raise ValueError(f"Unknown combination: variation={variation!r}, stage={stage}")
