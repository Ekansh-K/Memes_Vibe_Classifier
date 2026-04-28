"""Text preprocessing pipeline for MMHS150K tweet text and OCR text.

Transformation order (per coding standards):
1. Replace URLs → [URL]
2. Replace @mentions → [USER]
3. Segment hashtags using wordsegment
4. Normalize unicode (NFKC)
5. Collapse whitespace
6. Do NOT lowercase
7. Do NOT remove stopwords
"""

import re
import unicodedata
from functools import lru_cache
from typing import Optional

import wordsegment

# ── Load wordsegment n-gram data ONCE at module level ────────────────────────
wordsegment.load()

# ── Compiled regex patterns ──────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.co/\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")


# ── Hashtag segmentation ────────────────────────────────────────────────────

@lru_cache(maxsize=8192)
def segment_hashtag(tag: str) -> str:
    """Segment a hashtag body into space-separated words.

    Args:
        tag: Hashtag text WITHOUT the leading '#'.

    Returns:
        Space-separated segmented text, e.g. "DeportThemALL" → "Deport Them All"

    Short tags (<=3 chars) are returned as-is because segmentation is unreliable.
    wordsegment lowercases internally so we title-case the output.
    """
    if len(tag) <= 3:
        return tag
    segments = wordsegment.segment(tag)
    if not segments:
        return tag
    # Title-case each word; wordsegment lowercases everything internally
    return " ".join(w.capitalize() for w in segments)


def _replace_hashtags(text: str) -> str:
    """Replace each #hashtag with its segmented form."""

    def _repl(match: re.Match) -> str:
        body = match.group(1)
        return segment_hashtag(body)

    return _HASHTAG_RE.sub(_repl, text)


# ── Emoji handling (optional) ────────────────────────────────────────────────

def _convert_emojis_to_text(text: str) -> str:
    """Convert emoji characters to textual descriptions.

    Requires the `emoji` library. Falls back to no-op if not installed.
    Example: 😂 → ':face_with_tears_of_joy:'
    """
    try:
        import emoji

        return emoji.demojize(text, delimiters=(" :", ": "))
    except ImportError:
        return text


# ── Main cleaning functions ──────────────────────────────────────────────────

def clean_tweet_text(text: str, convert_emoji: bool = False) -> str:
    """Clean tweet text following the prescribed 5-step pipeline.

    Args:
        text: Raw tweet text from MMHS150K_GT.json.
        convert_emoji: If True, convert emojis to text descriptions.

    Returns:
        Cleaned text ready for tokenization.
    """
    if not text or not text.strip():
        return ""

    # Step 1: Replace URLs with [URL]
    text = _URL_RE.sub("[URL]", text)

    # Step 2: Replace @mentions with [USER]
    text = _MENTION_RE.sub("[USER]", text)

    # Step 3: Segment hashtags
    text = _replace_hashtags(text)

    # Step 4: Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Optional: Convert emojis to text
    if convert_emoji:
        text = _convert_emojis_to_text(text)

    # Step 5: Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text


def clean_ocr_text(text: str) -> str:
    """Clean OCR text — only unicode normalization and whitespace collapsing.

    OCR text is extracted from images, so URL/mention/hashtag processing
    is irrelevant.
    """
    if not text or not text.strip():
        return ""

    # Step 4: Normalize unicode (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Step 5: Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    return text
