"""P2 metrics — re-exports from existing modules + P2-specific end-to-end eval.

Reuses:
- src/evaluation/metrics.py: compute_binary_metrics, compute_multiclass_metrics
- src/p7/metrics.py: compute_multilabel_metrics, calibrate_thresholds,
                     apply_thresholds, compute_pipeline_metrics
"""

import logging

import numpy as np
from sklearn.metrics import f1_score

# Re-export from existing modules
from src.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics
from src.p7.metrics import (
    HATE_CAT_NAMES,
    apply_thresholds,
    calibrate_thresholds,
    compute_multilabel_metrics,
    compute_pipeline_metrics,
)

logger = logging.getLogger(__name__)

__all__ = [
    "HATE_CAT_NAMES",
    "apply_thresholds",
    "calibrate_thresholds",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "compute_multilabel_metrics",
    "compute_pipeline_metrics",
]
