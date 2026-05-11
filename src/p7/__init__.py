"""P7 — MHSDF (CNN + BiLSTM) Pipeline.

All P7-specific modules live here, keeping the pipeline self-contained
and separate from other tracks in the project.

Sub-modules:
    config    — P7Config dataclass for all 4 variations
    tokenizer — BERT-based tokenizer wrapper
    dataset   — P7-specific dataset (stage filtering, text-mode, multi-label)
    model     — MHSDF model (CNNVisualEncoder + BiLSTMTextEncoder)
    losses    — Loss factory (binary / multiclass / multilabel)
    trainer   — Two-stage training orchestrator
    evaluator — Evaluation harness + threshold calibration
    metrics   — Multi-label metrics (appended to existing evaluation)
"""
