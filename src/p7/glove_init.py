"""GloVe Twitter embedding initialisation for the BiLSTM text encoder.

Loads GloVe Twitter vectors and maps them to the RoBERTa/BERT tokenizer
vocabulary. Subword tokens that don't match any GloVe word are left at
random init (zero-mean Gaussian, std=0.02 — consistent with BERT init).

Usage:
    from src.p7.glove_init import build_glove_embedding_matrix
    weight = build_glove_embedding_matrix(
        glove_path="/path/to/glove.twitter.27B.200d.txt",
        tokenizer=tokenizer,
        embed_dim=200,
    )
    # Pass to MHSDF.from_config(..., pretrained_embeddings=weight)

Download GloVe Twitter:
    wget https://nlp.stanford.edu/data/glove.twitter.27B.zip
    unzip glove.twitter.27B.zip
    # Use glove.twitter.27B.200d.txt (200-dim, 1.2M vocab)
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _normalize_roberta_token(token: str) -> str:
    """Convert a RoBERTa BPE token to a lookup string for GloVe.

    RoBERTa uses 'Ġ' (U+0120) as a word-boundary prefix.
    Examples:
        'Ġhate'  → 'hate'
        'Ġthe'   → 'the'
        'kill'   → 'kill'     (no prefix = continuation subword, keep as-is)
        '</s>'   → ''          (special token, skip)
        '<pad>'  → ''          (special token, skip)
    """
    # Drop special tokens (angle-bracket wrapped or single-char control tokens)
    if token.startswith("<") and token.endswith(">"):
        return ""
    # Strip RoBERTa word-boundary marker
    token = token.replace("\u0120", "")
    # Lowercase and remove any remaining non-alphabetic noise
    token = token.lower().strip()
    return token


def build_glove_embedding_matrix(
    glove_path: str | Path,
    tokenizer,
    embed_dim: int = 200,
) -> torch.Tensor:
    """Build an embedding weight matrix initialised from GloVe Twitter.

    Args:
        glove_path: Path to the GloVe .txt file (e.g. glove.twitter.27B.200d.txt).
        tokenizer:  P7Tokenizer instance (its .tokenizer attribute gives the
                    HuggingFace fast tokenizer with .get_vocab()).
        embed_dim:  GloVe vector dimension (must match the file).

    Returns:
        FloatTensor of shape (vocab_size, embed_dim).
        Tokens that have no GloVe match are left at small random init.
    """
    glove_path = Path(glove_path)
    if not glove_path.exists():
        raise FileNotFoundError(
            f"[GloVe] File not found: {glove_path}\n"
            "Download: wget https://nlp.stanford.edu/data/glove.twitter.27B.zip"
        )

    # ── Load GloVe into a dict ───────────────────────────────────────────────
    logger.info(f"[GloVe] Loading vectors from {glove_path} ...")
    glove: dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            try:
                vec = np.array(parts[1:], dtype=np.float32)
                glove[word] = vec
            except ValueError:
                continue
    logger.info(f"[GloVe] Loaded {len(glove):,} vectors (dim={embed_dim}).")

    # ── Build embedding matrix ───────────────────────────────────────────────
    hf_tokenizer = tokenizer.tokenizer           # HuggingFace PreTrainedTokenizerFast
    vocab: dict[str, int] = hf_tokenizer.get_vocab()
    vocab_size = len(vocab)

    # Small random init for tokens with no GloVe match (std matches BERT init)
    weight = np.random.normal(0, 0.02, size=(vocab_size, embed_dim)).astype(np.float32)

    matched = 0
    for token, idx in vocab.items():
        word = _normalize_roberta_token(token)
        if word and word in glove:
            weight[idx] = glove[word]
            matched += 1

    coverage = matched / vocab_size * 100
    logger.info(
        f"[GloVe] Embedding matrix: {vocab_size:,} tokens, "
        f"{matched:,} matched ({coverage:.1f}% coverage)."
    )
    if coverage < 20:
        logger.warning(
            "[GloVe] Coverage < 20% — tokenizer vocab may not align well with GloVe. "
            "Consider using a word-level tokenizer for better GloVe compatibility."
        )

    return torch.tensor(weight, dtype=torch.float32)
