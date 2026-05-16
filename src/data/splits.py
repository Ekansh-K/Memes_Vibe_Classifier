"""Split loading, label computation, and dataset validation for MMHS150K.

Handles:
- Loading train/val/test split IDs from text files
- Loading and caching the ground-truth JSON
- Computing majority-vote, soft, and binary labels
- Generating and saving processed_labels.json
- Validating splits against GT and image directory
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from src.utils.config import (
    DATASET_DIR,
    GT_FILE,
    IMG_DIR,
    NUM_CLASSES_FINE,
    LABEL_MAP_FINE,
    OCR_CONSOLIDATED,
    OCR_CONSOLIDATED_FILTERED,
    OCR_DIR_OLD,
    OCR_DIR_NEW,
    PROCESSED_LABELS_FILE,
    SPLITS_DIR,
    TOTAL_SAMPLES,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
)

# ── Module-level caches ──────────────────────────────────────────────────────
_gt_cache: Optional[dict] = None
_labels_cache: Optional[dict] = None
_ocr_cache: dict = {}


# ── GT JSON loading ──────────────────────────────────────────────────────────

def load_gt_json(force_reload: bool = False) -> dict:
    """Load MMHS150K_GT.json and cache it in memory.

    Returns:
        Dict mapping tweet_id (str) to entry dict with keys:
        tweet_url, labels, img_url, tweet_text, labels_str
    """
    global _gt_cache
    if _gt_cache is not None and not force_reload:
        return _gt_cache
    with open(GT_FILE, "r", encoding="utf-8") as f:
        _gt_cache = json.load(f)
    return _gt_cache


# ── Split loading ────────────────────────────────────────────────────────────

def load_split_ids(split: str) -> list[str]:
    """Load tweet IDs for a given split.

    Args:
        split: One of "train", "val", "test".

    Returns:
        List of tweet ID strings.
    """
    valid_splits = {"train", "val", "test"}
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    path = SPLITS_DIR / f"{split}_ids.txt"
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


# ── Label computation ────────────────────────────────────────────────────────

def compute_majority_vote(labels: list[int]) -> int:
    """Compute majority vote from 3 annotator labels.

    Tie-breaking: prefer the lowest label ID (conservative bias toward
    NotHate=0 when annotators fully disagree). This is a deliberate design
    choice — when all 3 annotators differ, defaulting to 'not hate' is
    safer for the 11,701 fully-disagreed samples.
    """
    counter = Counter(labels)
    max_count = max(counter.values())
    candidates = [label for label, count in counter.items() if count == max_count]
    return min(candidates)


def compute_soft_labels(labels: list[int], num_classes: int = NUM_CLASSES_FINE) -> list[float]:
    """Convert annotator votes to a probability distribution.

    Examples:
        [0, 1, 0] → [0.667, 0.333, 0.0, 0.0, 0.0, 0.0]
        [1, 2, 5] → [0.0, 0.333, 0.333, 0.0, 0.0, 0.333]
    """
    soft = [0.0] * num_classes
    for label in labels:
        soft[label] += 1.0
    total = sum(soft)
    return [v / total for v in soft]


def to_binary(label: int) -> int:
    """Convert fine-grained label to binary. 0 (NotHate) → 0, else → 1."""
    return 0 if label == 0 else 1


def compute_agreement_level(labels: list[int]) -> int:
    """Compute annotator agreement level.

    Returns:
        3 if all agree, 2 if majority exists (2 of 3 agree), 1 if all differ.
    """
    unique = len(set(labels))
    if unique == 1:
        return 3
    elif unique == 2:
        return 2
    else:
        return 1


# ── Processed labels generation ──────────────────────────────────────────────

def generate_processed_labels(output_path: Optional[Path] = None) -> dict:
    """Compute and save processed labels for all samples.

    For each sample, produces:
        - hard_label_6class: int (majority vote)
        - hard_label_binary: int
        - soft_label_6class: list[float] (probability distribution)
        - soft_label_binary: list[float]
        - agreement_level: int (3, 2, or 1)
        - annotator_labels: list[int] (original 3 labels)

    Returns:
        Dict mapping tweet_id → label info dict.
    """
    gt = load_gt_json()
    processed = {}

    for tweet_id, entry in gt.items():
        labels = entry["labels"]
        hard_6 = compute_majority_vote(labels)
        hard_bin = to_binary(hard_6)
        soft_6 = compute_soft_labels(labels, NUM_CLASSES_FINE)
        # Soft binary: aggregate hate vs not-hate probabilities
        soft_bin = [soft_6[0], sum(soft_6[1:])]
        agreement = compute_agreement_level(labels)

        processed[tweet_id] = {
            "hard_label_6class": hard_6,
            "hard_label_binary": hard_bin,
            "soft_label_6class": soft_6,
            "soft_label_binary": soft_bin,
            "agreement_level": agreement,
            "annotator_labels": labels,
        }

    if output_path is None:
        output_path = PROCESSED_LABELS_FILE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed, f)
    print(f"[INFO] Saved processed labels for {len(processed)} samples → {output_path}")

    return processed


def load_processed_labels(force_reload: bool = False) -> dict:
    """Load processed labels from disk, generating if needed."""
    global _labels_cache
    if _labels_cache is not None and not force_reload:
        return _labels_cache

    if PROCESSED_LABELS_FILE.exists():
        with open(PROCESSED_LABELS_FILE, "r", encoding="utf-8") as f:
            _labels_cache = json.load(f)
    else:
        print("[INFO] processed_labels.json not found, generating...")
        _labels_cache = generate_processed_labels()

    return _labels_cache


# ── OCR data loading ─────────────────────────────────────────────────────────

def load_ocr_data(source: str = "filtered") -> dict:
    """Load OCR text data.

    Args:
        source: "filtered" (default – cleaned ocr_filtered.json, no phone UI noise),
                "new"      (raw ocr_consolidated.json, includes phone UI text),
                "old"      (original per-image img_txt/ JSONs, 2018 quality),
                "both"     (merge old+new, new takes priority).

    Returns:
        Dict mapping tweet_id → ocr text string.
    """
    if source in _ocr_cache:
        return _ocr_cache[source]

    result = {}

    # ── Filtered (default) ────────────────────────────────────────────────
    if source == "filtered":
        if OCR_CONSOLIDATED_FILTERED.exists():
            with open(OCR_CONSOLIDATED_FILTERED, "r", encoding="utf-8") as f:
                data = json.load(f)
            result = {k: v for k, v in data.items() if v and v.strip()}
        else:
            import warnings
            warnings.warn(
                f"ocr_filtered.json not found at {OCR_CONSOLIDATED_FILTERED}. "
                "Falling back to ocr_consolidated.json (unfiltered). "
                "Run scripts/filter_ocr.py to create ocr_filtered.json.",
                stacklevel=2,
            )
            if OCR_CONSOLIDATED.exists():
                with open(OCR_CONSOLIDATED, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result = {k: v for k, v in data.items() if v and v.strip()}

    # ── Old per-image JSONs ───────────────────────────────────────────────
    if source in ("old", "both"):
        if OCR_DIR_OLD.exists():
            for p in OCR_DIR_OLD.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = data.get("img_text", "").strip()
                    if text:
                        result[p.stem] = text
                except (json.JSONDecodeError, KeyError):
                    continue

    # ── New consolidated (unfiltered) ─────────────────────────────────────
    if source in ("new", "both"):
        if OCR_CONSOLIDATED.exists():
            with open(OCR_CONSOLIDATED, "r", encoding="utf-8") as f:
                new_data = json.load(f)
            # new takes priority when source="both"
            result.update({k: v for k, v in new_data.items() if v.strip()})
        elif OCR_DIR_NEW.exists():
            for p in OCR_DIR_NEW.glob("*.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    text = data.get("ocr_text", "").strip()
                    if text:
                        result[p.stem] = text
                except (json.JSONDecodeError, KeyError):
                    continue

    _ocr_cache[source] = result
    return result


# ── Validation ───────────────────────────────────────────────────────────────

def validate_splits() -> dict:
    """Run all validation checks on splits and dataset integrity.

    Returns:
        Dict with validation results and statistics.
    """
    gt = load_gt_json()
    gt_ids = set(gt.keys())

    train_ids = set(load_split_ids("train"))
    val_ids = set(load_split_ids("val"))
    test_ids = set(load_split_ids("test"))

    # Check sizes
    results = {
        "train_size": len(train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "gt_size": len(gt_ids),
    }

    # Check no overlap
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    results["overlaps"] = {
        "train_val": len(train_val_overlap),
        "train_test": len(train_test_overlap),
        "val_test": len(val_test_overlap),
    }
    assert len(train_val_overlap) == 0, f"Train/val overlap: {len(train_val_overlap)}"
    assert len(train_test_overlap) == 0, f"Train/test overlap: {len(train_test_overlap)}"
    assert len(val_test_overlap) == 0, f"Val/test overlap: {len(val_test_overlap)}"

    # Check all IDs exist in GT
    all_split_ids = train_ids | val_ids | test_ids
    missing_from_gt = all_split_ids - gt_ids
    extra_in_gt = gt_ids - all_split_ids
    results["missing_from_gt"] = len(missing_from_gt)
    results["extra_in_gt"] = len(extra_in_gt)

    # Check images exist
    if IMG_DIR.exists():
        img_files = {p.stem for p in IMG_DIR.glob("*.jpg")}
        missing_images = all_split_ids - img_files
        results["missing_images"] = len(missing_images)
    else:
        results["missing_images"] = -1  # directory not found

    # Label distribution per split
    labels = load_processed_labels()
    for split_name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        dist = Counter()
        for sid in split_ids:
            if sid in labels:
                dist[labels[sid]["hard_label_6class"]] += 1
        results[f"{split_name}_label_dist"] = {
            LABEL_MAP_FINE.get(k, str(k)): v for k, v in sorted(dist.items())
        }

    # OCR coverage
    ocr_old_ids = {p.stem for p in OCR_DIR_OLD.glob("*.json")} if OCR_DIR_OLD.exists() else set()
    results["ocr_old_coverage"] = len(ocr_old_ids & all_split_ids)

    # Agreement stats
    agreement_counts = Counter()
    for sid in all_split_ids:
        if sid in labels:
            agreement_counts[labels[sid]["agreement_level"]] += 1
    results["agreement_distribution"] = dict(sorted(agreement_counts.items()))

    print(f"[VALIDATION] Train: {results['train_size']}, Val: {results['val_size']}, Test: {results['test_size']}")
    print(f"[VALIDATION] No split overlaps: ✓")
    print(f"[VALIDATION] Missing from GT: {results['missing_from_gt']}, Extra in GT: {results['extra_in_gt']}")
    print(f"[VALIDATION] Missing images: {results['missing_images']}")
    print(f"[VALIDATION] Old OCR coverage: {results['ocr_old_coverage']}/{len(all_split_ids)}")
    print(f"[VALIDATION] Agreement dist: {results['agreement_distribution']}")

    return results


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate processed labels and validate splits")
    parser.add_argument("--generate-labels", action="store_true", help="Generate processed_labels.json")
    parser.add_argument("--validate", action="store_true", help="Run split validation checks")
    args = parser.parse_args()

    if args.generate_labels:
        generate_processed_labels()

    if args.validate:
        validate_splits()

    if not args.generate_labels and not args.validate:
        # Default: do both
        generate_processed_labels()
        validate_splits()
