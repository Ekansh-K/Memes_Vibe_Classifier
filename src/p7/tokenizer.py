"""BERT Tokenizer wrapper for P7 (MHSDF).

Uses Hugging Face AutoTokenizer with cardiffnlp/twitter-roberta-base
(optimised for tweet-style text including memes, emojis, hashtags).

Falls back to bert-base-uncased if the twitter-roberta model is unavailable.
"""

import logging
from typing import Union

import torch
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Preferred model — trained on ~58M tweets, handles meme text well
_PRIMARY_MODEL = "cardiffnlp/twitter-roberta-base"
_FALLBACK_MODEL = "bert-base-uncased"

# Module-level tokenizer cache (shared across all P7 instances in a process)
_tokenizer_cache: dict[str, AutoTokenizer] = {}


def get_tokenizer(model_name: str = _PRIMARY_MODEL) -> AutoTokenizer:
    """Load and cache a Hugging Face tokenizer.

    Tries model_name first; falls back to bert-base-uncased if loading fails.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded AutoTokenizer instance (cached after first call).
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"[P7] Loaded tokenizer: {model_name}  (vocab size={tokenizer.vocab_size})")
    except Exception as e:
        logger.warning(
            f"[P7] Could not load '{model_name}': {e}. "
            f"Falling back to '{_FALLBACK_MODEL}'."
        )
        tokenizer = AutoTokenizer.from_pretrained(_FALLBACK_MODEL)
        logger.info(f"[P7] Loaded fallback tokenizer: {_FALLBACK_MODEL}")

    _tokenizer_cache[model_name] = tokenizer
    return tokenizer


class P7Tokenizer:
    """Thin wrapper around a Hugging Face tokenizer for P7's BiLSTM.

    Encodes a single text or a batch to fixed-length token ID tensors.
    The BiLSTM uses token IDs + a learned embedding layer, NOT BERT's
    contextual representations — so we only need the tokenizer's vocabulary,
    not the BERT encoder weights.

    Usage:
        tok = P7Tokenizer(max_seq_len=128)
        ids = tok.encode("This is a meme caption.")        # list[int]
        batch = tok.encode_batch(["text1", "text2"])       # Tensor (2, 128)
    """

    def __init__(
        self,
        model_name: str = _PRIMARY_MODEL,
        max_seq_len: int = 128,
    ):
        self.max_seq_len = max_seq_len
        self.model_name = model_name
        self._tok = get_tokenizer(model_name)
        self.tokenizer = self._tok          # public alias used by glove_init.py
        self.vocab_size: int = self._tok.vocab_size
        self.pad_id: int = self._tok.pad_token_id or 0

    def encode(self, text: str) -> list[int]:
        """Encode a single text string to a padded list of token IDs.

        Args:
            text: Input text string.

        Returns:
            List of int token IDs, length == max_seq_len.
        """
        if not text or not text.strip():
            return [self.pad_id] * self.max_seq_len

        ids = self._tok.encode(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        return ids

    def encode_batch(self, texts: list[str]) -> torch.LongTensor:
        """Encode a batch of text strings.

        Args:
            texts: List of B text strings.

        Returns:
            LongTensor of shape (B, max_seq_len).
        """
        # Handle empty/None texts gracefully
        cleaned = [t if (t and t.strip()) else "" for t in texts]

        enc = self._tok(
            cleaned,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )
        return enc["input_ids"]  # (B, max_seq_len)

    def __repr__(self) -> str:
        return (
            f"P7Tokenizer(model={self.model_name!r}, "
            f"vocab_size={self.vocab_size}, max_seq_len={self.max_seq_len})"
        )
