"""P2 — TCAM (Text-guided Cross-Attention Multimodal) pipeline.

Architecture: Frozen CLIP ViT-L/14 + Frozen TweetEval RoBERTa
             → learnable cross-attention fusion → classification head.

Variations:
    A — Binary-only (Stage 1 standalone)
    B — Direct 6-class multi-class (single stage)
    C — Two-stage hierarchical, single-label Stage 2
    D — Two-stage hierarchical, multi-label Stage 2 (primary target)
"""

from src.p2.config import P2Config
from src.p2.model import TCAM
from src.p2.trainer import run_p2

__all__ = ["P2Config", "TCAM", "run_p2"]
