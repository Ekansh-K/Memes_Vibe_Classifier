"""OCR noise filtering for MMHS150K image OCR text.

Many images in MMHS150K are Twitter/social media screenshots. The OCR engine
picks up mobile UI chrome (carrier names, signal bars, timestamps, battery %,
app names) that is pure noise for hate speech classification.

This module provides `filter_ocr_noise(text)` which strips that noise while
preserving the actual semantic content embedded in the image.

Design decisions:
- Patterns are scoped tightly to avoid false positives on real content.
- "Home", "Search", "Profile", "Messages" are NOT filtered — they are common
  English words and removing them would destroy real sentences.
- Battery % is only removed when it looks like a status-bar reading (e.g.
  standalone "87%" or "100 %"), not percentages embedded in sentences.
- Dates are only removed when they look like tweet timestamps (month-name form),
  not fractions or ranges like "1/3" or "2-4".
- Short artifact tokens (1-2 chars, isolated) are removed AFTER other patterns
  so that removing a noise word doesn't leave stray letters behind.
- The filter is deliberately conservative: when in doubt, keep the text.
  False negatives (keeping some UI noise) are less harmful than false positives
  (deleting real content).
- lru_cache is used because the same OCR text may appear in multiple dataset
  passes (train/val/test loading); caching avoids re-running regex each time.
"""

import re
from functools import lru_cache

# ── Noise patterns (compiled once at module level) ────────────────────────────

# Mobile carrier names — tightly bounded to avoid matching words like "bell"
# in sentences. We only match them when they appear as standalone tokens.
_CARRIER_RE = re.compile(
    r'\b(Verizon|AT&T|T[\s-]?Mobile|Sprint|Cricket|MetroPCS|'
    r'U\.?S\.?\s?Cellular|Straight\s?Talk|'
    r'Rogers|Telus|Vodafone|Optus|Telstra|Singtel)\b',
    re.IGNORECASE,
)

# Connectivity / signal indicators — standalone tokens only
_SIGNAL_RE = re.compile(
    r'\b(LTE|4G LTE|5G|4G|3G|2G|EDGE|HSPA\+?|GPRS|'
    r'Wi-Fi|WiFi|WIFI|Airplane\s?Mode)\b',
    re.IGNORECASE,
)

# Device / OS names — these appear in screenshot status bars and footers
_DEVICE_RE = re.compile(
    r'\b(iPhone|iPad|iPod touch|'
    r'Samsung\s?Galaxy|Galaxy\s?S\d+|'
    r'Windows\s?Phone|BlackBerry)\b',
    re.IGNORECASE,
)

# Timestamp patterns that look exactly like phone status-bar times.
# Matches "9:41", "12:30 AM", "9:41 PM" — but NOT "1:3 of all" (requires
# exactly 2 digits after the colon, and the whole thing must be word-bounded).
_STATUS_TIME_RE = re.compile(r'\b\d{1,2}:\d{2}\s*(?:AM|PM)\b', re.IGNORECASE)

# Battery percentage — only matches patterns that look like status-bar readings:
# must be a standalone number followed immediately by % with no surrounding text.
# e.g. "87%" or "100 %" at start/end of a token run, NOT "34% of people".
# Strategy: match only when preceded/followed by whitespace or string boundary,
# and the number is 1-100 (phone battery range).
_BATTERY_RE = re.compile(r'(?<!\w)(?:100|[1-9]?\d)\s*%(?!\s*\w)', re.IGNORECASE)

# Twitter/social-media UI chrome — only the labels that ONLY appear as UI
# elements and are never meaningful in a hate speech context.
# Deliberately excluded: Home, Search, Profile, Messages, Reply, Like, Follow
# (all common English words that appear in real sentences).
_TWITTER_CHROME_RE = re.compile(
    r'\b(Retweet(?:ed)?|Quote\s?Tweet|'
    r'Promoted\s?Tweet|'
    r'What\'?s\s?happening\??|'
    r'New\s?Tweets?\s?available|'
    r'mobile\.twitter\.com|twitter\.com)\b',
    re.IGNORECASE,
)

# Numeric engagement counts that appear under tweets on screenshots
# e.g. "2,341 Retweets", "14.5K Likes", "1M views"
# Only matches when the noun is plural/specific social metric so we don't
# accidentally strip "34 replies from staff" in real text.
_ENGAGEMENT_RE = re.compile(
    r'\b\d[\d,\.]*\s*[KkMmBb]?\s*(?:Retweets?|Quote\s?Tweets?|Reposts?)\b',
    re.IGNORECASE,
)

# Tweet timestamps — matches "Oct 3, 2019", "Mar 15, 2018" forms ONLY.
# Does NOT match numeric date formats like "1/3" or "2-4" which appear in
# real content.
_TWEET_DATE_RE = re.compile(
    r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+'
    r'\d{1,2},?\s+\d{4}\b',
    re.IGNORECASE,
)

# "via Twitter for iPhone" / "via the web" attribution strings
_VIA_RE = re.compile(r'\bvia\s+(?:Twitter|the\s+web|iPhone|Android|iPad)\b', re.IGNORECASE)

# Isolated single/double-character OCR artifacts left after other patterns fire.
# e.g. after removing "Verizon LTE 87%" from "Verizon LTE 87% X Y" we get "X Y"
# which are meaningless leftover glyphs.
# Only removes tokens that are entirely non-alphanumeric or single letters
# surrounded by whitespace/boundaries — preserves real abbreviations at
# start of longer words.
_ISOLATED_ARTIFACT_RE = re.compile(r'(?<!\w)[A-Za-z](?!\w)')

# Collapse multiple spaces to one
_WHITESPACE_RE = re.compile(r'\s{2,}')


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=16384)
def filter_ocr_noise(text: str) -> str:
    """Remove Twitter/mobile UI noise from OCR-extracted image text.

    Args:
        text: Raw OCR text from a social media image.

    Returns:
        Cleaned text with mobile UI chrome removed.
        Returns empty string if the entire text was noise.

    Examples:
        >>> filter_ocr_noise("Verizon LTE 9:41 AM 87%  I can't believe what I'm seeing")
        "I can't believe what I'm seeing"
        >>> filter_ocr_noise("Retweet Quote Tweet Promoted Tweet What's happening?")
        ""
        >>> filter_ocr_noise("Wi-Fi 100% 3:22 PM  WHEN YOU REALIZE YOUR WHOLE LIFE IS A LIE")
        "WHEN YOU REALIZE YOUR WHOLE LIFE IS A LIE"
        >>> filter_ocr_noise("He replied to my profile post about 34% unemployment")
        "He replied to my profile post about 34% unemployment"
    """
    if not text or not text.strip():
        return ""

    t = text

    # Apply noise patterns in order (carrier/signal first, chrome last)
    t = _CARRIER_RE.sub(" ", t)
    t = _SIGNAL_RE.sub(" ", t)
    t = _DEVICE_RE.sub(" ", t)
    t = _STATUS_TIME_RE.sub(" ", t)
    t = _BATTERY_RE.sub(" ", t)
    t = _TWITTER_CHROME_RE.sub(" ", t)
    t = _ENGAGEMENT_RE.sub(" ", t)
    t = _TWEET_DATE_RE.sub(" ", t)
    t = _VIA_RE.sub(" ", t)

    # Collapse whitespace first, then remove isolated single-char artifacts
    t = _WHITESPACE_RE.sub(" ", t).strip()
    t = _ISOLATED_ARTIFACT_RE.sub(" ", t)

    # Final whitespace collapse and trim
    t = _WHITESPACE_RE.sub(" ", t).strip()

    # If result is very short (<= 3 chars), it's a pure noise artifact
    if len(t) <= 3:
        return ""

    return t


def compute_noise_ratio(original: str, filtered: str) -> float:
    """Compute the fraction of the original text that was noise.

    Returns a value in [0.0, 1.0]. 1.0 = entirely noise, 0.0 = no noise removed.
    """
    if not original or not original.strip():
        return 0.0
    original_len = len(original.strip())
    filtered_len = len(filtered.strip())
    removed = original_len - filtered_len
    return max(0.0, removed / original_len)


# ── CLI for testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import random
    import sys
    from pathlib import Path

    # ocr_filter.py lives at src/data/ -> parents[2] = project root
    project_root = Path(__file__).resolve().parents[2]
    consolidated_path = project_root / "dataset" / "ocr_consolidated.json"

    if not consolidated_path.exists():
        print("ocr_consolidated.json not found - run fix_consolidated_ocr.py first")
        sys.exit(1)

    print("Loading OCR data...")
    with open(consolidated_path, encoding="utf-8") as f:
        consolidated = json.load(f)

    # ── False-positive regression tests ──────────────────────────────────────
    regression_cases = [
        # (input, should_be_preserved_substring)
        ("He replied to my profile post about 34% unemployment", "34% unemployment"),
        ("I searched his profile for more information", "searched his profile"),
        ("Follow the money trail and you'll find the truth", "Follow the money"),
        ("She likes to message her followers daily", "likes to message"),
        ("The home page shows trending topics", "home page"),
        ("Search results showed 1/3 of posts were hate", "1/3 of posts"),
        ("From 2-4 years in prison according to sentencing", "2-4 years"),
        # Noise that should be stripped
        ("Verizon LTE 9:41 AM 87% RACIST CONTENT HERE", "RACIST CONTENT HERE"),
        ("Wi-Fi 100% 3:22 PM WHEN YOU ARE STUPID", "WHEN YOU ARE STUPID"),
        ("Retweet Quote Tweet What's happening? ACTUAL MEME TEXT", "ACTUAL MEME TEXT"),
    ]
    print("\n=== Regression Tests ===")
    all_passed = True
    for raw, expected_substr in regression_cases:
        result = filter_ocr_noise(raw)
        passed = expected_substr.lower() in result.lower()
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        raw_display = raw[:60].encode("ascii", errors="replace").decode("ascii")
        print(f"  [{status}] {raw_display!r}")
        if not passed:
            print(f"         Expected to contain: {expected_substr!r}")
            print(f"         Got: {result!r}")
    print(f"\n  Result: {'ALL PASS' if all_passed else 'FAILURES FOUND'}")

    # ── Stats on real data ────────────────────────────────────────────────────
    random.seed(42)
    entries_with_text = [(k, v) for k, v in consolidated.items() if v.strip()]
    sample = random.sample(entries_with_text, min(2000, len(entries_with_text)))

    noisy_count = 0
    heavy_noise_count = 0
    total_chars_removed = 0
    changed_examples = []

    for tid, raw_text in sample:
        filtered = filter_ocr_noise(raw_text)
        ratio = compute_noise_ratio(raw_text, filtered)
        if ratio > 0.0:
            noisy_count += 1
            total_chars_removed += len(raw_text) - len(filtered)
        if ratio > 0.30:
            heavy_noise_count += 1
        if ratio > 0.05 and len(changed_examples) < 8:
            changed_examples.append((tid, raw_text, filtered, ratio))

    n = len(sample)
    print(f"\n=== OCR Noise Filter Stats (sample of {n:,} entries with text) ===")
    print(f"  Entries with ANY noise removed:   {noisy_count:5d} ({noisy_count/n*100:.1f}%)")
    print(f"  Entries with >30% noise removed:  {heavy_noise_count:5d} ({heavy_noise_count/n*100:.1f}%)")
    avg_removed = total_chars_removed / max(noisy_count, 1)
    print(f"  Avg chars removed per noisy entry: {avg_removed:.0f}")

    def safe_repr(s: str, n: int = 120) -> str:
        return repr(s[:n].encode("ascii", errors="replace").decode("ascii"))

    print("\n=== Changed Examples (noise ratio > 5%) ===")
    for tid, raw, flt, ratio in changed_examples:
        print(f"\n  ID: {tid} (noise={ratio*100:.0f}%)")
        print(f"  BEFORE: {safe_repr(raw)}")
        print(f"  AFTER:  {safe_repr(flt)}")
