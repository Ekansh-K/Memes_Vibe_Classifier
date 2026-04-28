"""Generate VLM instruction-tuning data for Track B (Qwen2.5-VL / LLaVA).

Creates JSONL files in the LLaVA conversation format with:
  - Binary classification variant (hateful / not hateful)
  - 6-class variant (fine-grained label)

Usage:
    python scripts/generate_vlm_data.py
    python scripts/generate_vlm_data.py --mode binary --split train
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import clean_ocr_text, clean_tweet_text
from src.data.splits import load_gt_json, load_ocr_data, load_processed_labels, load_split_ids
from src.utils.config import DATASET_DIR, LABEL_MAP_FINE


def build_prompt(tweet_text: str, ocr_text: str, mode: str = "binary") -> str:
    """Build the instruction prompt for a VLM.

    Args:
        tweet_text: Preprocessed tweet text.
        ocr_text: Preprocessed OCR text from image (may be empty).
        mode: "binary" or "multiclass".

    Returns:
        Formatted prompt string.
    """
    parts = ["<image>"]
    parts.append(f'Tweet: "{tweet_text}"')

    if ocr_text:
        parts.append(f'Text visible in image: "{ocr_text}"')

    if mode == "binary":
        parts.append(
            "\nAnalyze this tweet and associated image for hate speech. "
            "Consider both textual content and visual context. "
            "Is this hateful or not hateful?"
        )
    else:
        parts.append(
            "\nAnalyze this tweet and associated image for hate speech. "
            "Consider both textual content and visual context. "
            "Classify into one of: NotHate, Racist, Sexist, Homophobe, Religion, OtherHate."
        )

    return "\n".join(parts)


def build_answer(label_info: dict, mode: str = "binary") -> str:
    """Build the expected answer string."""
    if mode == "binary":
        return "not hateful" if label_info["hard_label_binary"] == 0 else "hateful"
    else:
        label_id = label_info["hard_label_6class"]
        return LABEL_MAP_FINE[label_id]


def generate_vlm_data(
    split: str,
    mode: str = "binary",
    ocr_source: str = "old",
    convert_emoji: bool = False,
) -> list[dict]:
    """Generate VLM-format data for a single split.

    Returns list of conversation dicts in LLaVA format.
    """
    gt = load_gt_json()
    labels = load_processed_labels()
    ocr_data = load_ocr_data(ocr_source)
    split_ids = load_split_ids(split)

    data = []
    for tweet_id in split_ids:
        if tweet_id not in gt or tweet_id not in labels:
            continue

        entry = gt[tweet_id]
        label_info = labels[tweet_id]

        tweet_text = clean_tweet_text(entry["tweet_text"], convert_emoji=convert_emoji)
        ocr_text = clean_ocr_text(ocr_data.get(tweet_id, ""))

        prompt = build_prompt(tweet_text, ocr_text, mode=mode)
        answer = build_answer(label_info, mode=mode)

        # Relative image path from project root
        img_rel = f"dataset/img_resized/{tweet_id}.jpg"

        sample = {
            "id": tweet_id,
            "image": img_rel,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ],
        }
        data.append(sample)

    return data


def save_jsonl(data: list[dict], output_path: Path) -> None:
    """Save data as JSON-lines file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved {len(data)} samples → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate VLM instruction-tuning data")
    parser.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--split", type=str, default=None, help="Specific split or all")
    parser.add_argument("--ocr-source", default="old", choices=["old", "new", "both"])
    parser.add_argument("--output-dir", type=str, default=str(DATASET_DIR))
    args = parser.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]
    output_dir = Path(args.output_dir)

    for split in splits:
        data = generate_vlm_data(split=split, mode=args.mode, ocr_source=args.ocr_source)
        filename = f"vlm_{split}_{args.mode}.jsonl"
        save_jsonl(data, output_dir / filename)


if __name__ == "__main__":
    main()
