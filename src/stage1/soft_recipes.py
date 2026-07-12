"""Stage-1 soft-label training recipes (S0–S7).

See plan: soft labels often underperform when pos_weight + aggressive
agreement downweighting cancel the soft signal. These recipes ablate that.

Recipes
-------
S0  Hard labels + pos_weight (control; classic imbalanced BCE)
S1  Soft BCE, no pos_weight
S2  Soft BCE + pos_weight (legacy / previous default)
S3  Soft BCE + gentler binary agreement weights (0.7, 0.9, 1.0)
S4  Soft BCE, no agreement weighting
S5  Multi-task: hard BCE(pos_weight) + λ soft BCE (no pos_weight)
S6  Soft BCE + filter train to agreement_binary >= min_agreement_binary
S7  Soft BCE + soft-target temperature sharpening
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.p2.losses import SoftBCEWithAgreementWeighting, compute_binary_pos_weight

SOFT_RECIPES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


@dataclass
class SoftRecipeSpec:
    name: str
    use_soft_labels: bool
    use_pos_weight: bool
    use_agreement_weighting: bool
    agreement_weights: tuple
    use_binary_agreement: bool
    multi_task: bool
    multi_task_lambda: float
    soft_temperature: float  # 1.0 = no sharpen; >1 flattens, <1 sharpens
    min_agreement_binary: Optional[int]  # filter train if set (S6)
    focal_gamma: float  # 0 = off


def get_soft_recipe(name: str) -> SoftRecipeSpec:
    key = (name or "S2").upper().strip()
    if key not in SOFT_RECIPES:
        raise ValueError(f"Unknown soft_recipe={name!r}. Choose from {SOFT_RECIPES}")

    # Defaults resemble legacy S2
    base = dict(
        name=key,
        use_soft_labels=True,
        use_pos_weight=True,
        use_agreement_weighting=True,
        agreement_weights=(0.4, 0.7, 1.0),
        use_binary_agreement=True,
        multi_task=False,
        multi_task_lambda=0.5,
        soft_temperature=1.0,
        min_agreement_binary=None,
        focal_gamma=0.0,
    )

    if key == "S0":
        base.update(use_soft_labels=False, use_pos_weight=True, use_agreement_weighting=False)
    elif key == "S1":
        base.update(use_pos_weight=False, use_agreement_weighting=False)
    elif key == "S2":
        pass  # legacy
    elif key == "S3":
        base.update(
            use_pos_weight=False,
            agreement_weights=(0.7, 0.9, 1.0),
            use_agreement_weighting=True,
        )
    elif key == "S4":
        base.update(use_pos_weight=False, use_agreement_weighting=False)
    elif key == "S5":
        base.update(
            use_soft_labels=True,  # multi-task uses both
            use_pos_weight=True,  # for hard branch
            use_agreement_weighting=False,
            multi_task=True,
            multi_task_lambda=0.5,
        )
    elif key == "S6":
        base.update(
            use_pos_weight=False,
            use_agreement_weighting=True,
            agreement_weights=(0.7, 0.9, 1.0),
            min_agreement_binary=2,
        )
    elif key == "S7":
        base.update(
            use_pos_weight=False,
            use_agreement_weighting=True,
            agreement_weights=(0.7, 0.9, 1.0),
            soft_temperature=0.7,  # sharpen toward 0/1
        )

    return SoftRecipeSpec(**base)


def apply_soft_temperature(soft: torch.Tensor, temperature: float) -> torch.Tensor:
    """Sharpen/flatten soft hate probs in (0,1). temperature=1 identity.

    Uses logit-space: sigmoid(logit(p) / T). T<1 sharpens, T>1 flattens.
    """
    if temperature is None or abs(temperature - 1.0) < 1e-6:
        return soft
    p = soft.clamp(1e-4, 1.0 - 1e-4)
    logit = torch.log(p) - torch.log1p(-p)
    return torch.sigmoid(logit / temperature)


def build_stage1_criterion(
    recipe: SoftRecipeSpec,
    hard_labels: list[int],
    device: torch.device,
) -> SoftBCEWithAgreementWeighting:
    """Build primary SoftBCE criterion for the recipe."""
    if recipe.use_pos_weight and not recipe.multi_task:
        pw = compute_binary_pos_weight(hard_labels, device)
    elif recipe.multi_task:
        # Soft branch of multi-task uses no pos_weight; hard branch separate
        pw = None
    else:
        pw = None

    # Multi-task: primary criterion is soft without pos_weight
    if recipe.multi_task:
        pw = None

    return SoftBCEWithAgreementWeighting(
        pos_weight=pw,
        agreement_weights=recipe.agreement_weights,
        use_agreement_weighting=recipe.use_agreement_weighting,
        label_smoothing=0.0,
        focal_gamma=recipe.focal_gamma,
    ).to(device)


def build_hard_criterion(
    hard_labels: list[int],
    device: torch.device,
) -> SoftBCEWithAgreementWeighting:
    """Hard BCE with pos_weight (for multi-task S5 hard branch)."""
    pw = compute_binary_pos_weight(hard_labels, device)
    return SoftBCEWithAgreementWeighting(
        pos_weight=pw,
        use_agreement_weighting=False,
        label_smoothing=0.0,
    ).to(device)
