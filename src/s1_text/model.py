"""Full fine-tune transformer for binary hate (Stage 1)."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class TextHateClassifier(nn.Module):
    """HF encoder + linear head → single logit (hate). Entire encoder trainable."""

    def __init__(
        self,
        model_name: str,
        dropout: float = 0.2,
        max_seq_len: int = 192,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        # Train everything
        for p in self.encoder.parameters():
            p.requires_grad = True
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"[S1Text] {model_name}  trainable_params={n_train:,}")

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, texts: list[str] | None = None, **token_batch) -> torch.Tensor:
        """Returns logits (B,)."""
        if texts is not None:
            device = next(self.parameters()).device
            token_batch = self.tokenize(texts, device)
        out = self.encoder(**token_batch)
        # Prefer pooler if present; else CLS
        if getattr(out, "pooler_output", None) is not None and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            pooled = out.last_hidden_state[:, 0, :]
        return self.head(self.dropout(pooled)).squeeze(-1)
