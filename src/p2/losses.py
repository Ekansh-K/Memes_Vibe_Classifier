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


class SoftBCEWithAgreementWeighting(nn.Module):
    """BCEWithLogitsLoss that supports soft targets, agreement weighting,
    and BCE-specific label smoothing (Options A + B + C combined).

    For each sample i:
        smoothed_target = target * (1 - label_smoothing) + 0.5 * label_smoothing
        per_sample_loss = BCE(logit, smoothed_target) * agreement_weight[agreement_level]
        loss = weighted_mean(per_sample_loss)
    """

    def __init__(
        self,
        pos_weight: torch.Tensor,
        agreement_weights: tuple = (0.2, 0.5, 1.0),
        use_agreement_weighting: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # reduction='none' for per-sample weighting
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        self.use_agreement_weighting = use_agreement_weighting
        # agreement_weights: index 0 = level 1 (all differ), 1 = level 2, 2 = level 3
        self.register_buffer(
            "_aw",
            torch.tensor([agreement_weights[0], agreement_weights[1], agreement_weights[2]],
                         dtype=torch.float32),
        )
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        agreement_levels: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute weighted soft-BCE loss.

        Args:
            logits:           (B,) raw logit scores
            targets:          (B,) soft targets in [0, 1] (annotator vote probs)
            agreement_levels: (B,) int tensor with values 1, 2, or 3
        """
        # Option C: BCE label smoothing — push targets toward 0.5
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        per_sample = self.bce(logits, targets)  # (B,)

        # Option B: agreement-based per-sample weighting
        if self.use_agreement_weighting and agreement_levels is not None:
            # Map agreement levels (1,2,3) to weights via index (0,1,2).
            # Ensure same device — _aw follows the module, agreement_levels follows the batch.
            w = self._aw[(agreement_levels - 1).to(self._aw.device)]  # (B,)
            w = w.to(per_sample.device)  # move weight to loss device for multiply
            # Weighted mean (normalize by sum of weights for stable gradients)
            return (per_sample * w).sum() / w.sum().clamp(min=1.0)

        return per_sample.mean()


class TemperatureScaler(nn.Module):
    """Learn a single temperature parameter to calibrate output probabilities.

    After training, divide logits by learned temperature before sigmoid.
    This stabilizes the threshold and calibrates the output distribution.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 200):
        """Fit temperature on validation logits using NLL loss.

        Args:
            logits: (N,) raw validation logits
            labels: (N,) binary labels (hard, 0 or 1)
            lr:     learning rate for LBFGS
            max_iter: maximum LBFGS iterations
        """
        self.temperature.requires_grad_(True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = criterion(scaled, labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad_(False)
        logger.info(
            f"[TempScale] Learned temperature = {self.temperature.item():.4f}"
        )
        return self.temperature.item()


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
    config=None,
):
    """Return the appropriate loss function for the given P2 variant and stage.

    Args:
        variation:      "A", "B", "C", or "D".
        stage:          1 or 2 (for C/D); ignored for A/B (use stage=1).
        train_dataset:  P2Dataset instance for the training split.
        device:         Target device.
        label_smoothing: Amount of label smoothing.
        config:         P2Config (needed for soft labels / agreement weighting).

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

        # Use soft labels + agreement weighting if config enables them
        use_soft = getattr(config, "use_soft_labels", False)
        use_agree = getattr(config, "use_agreement_weighting", False)
        ls = label_smoothing if label_smoothing > 0 else 0.0

        if use_soft or use_agree or ls > 0:
            agree_w = getattr(config, "agreement_weights", (0.2, 0.5, 1.0))
            logger.info(
                f"[P2 Loss] Stage 1: soft_labels={use_soft}  "
                f"agreement_weighting={use_agree} ({agree_w})  "
                f"label_smoothing={ls}"
            )
            return SoftBCEWithAgreementWeighting(
                pos_weight=pw,
                agreement_weights=agree_w,
                use_agreement_weighting=use_agree,
                label_smoothing=ls,
            )

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
