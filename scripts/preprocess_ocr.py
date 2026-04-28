"""OCR re-extraction script for MMHS150K images using PaddleOCR 3.x.

Processes all images in dataset/img_resized/ and saves:
  - Per-image JSON → dataset/img_txt_new/{tweet_id}.json
  - Consolidated mapping → dataset/ocr_consolidated.json
  - Failure log → dataset/ocr_failures.txt

Supports --resume to skip already-processed images.
Falls back to EasyOCR if PaddleOCR is unavailable.

Usage:
    python scripts/preprocess_ocr.py --device gpu:0
    python scripts/preprocess_ocr.py --device cpu --engine easyocr
    python scripts/preprocess_ocr.py --resume  # continue after crash
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# PaddleOCR env setup: skip slow connectivity check on import
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DATASET_DIR, IMG_DIR, OCR_CONSOLIDATED, OCR_DIR_NEW


def get_image_paths(img_dir: Path) -> list[Path]:
    """Get all .jpg image paths sorted by name."""
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        # Also try .png
        paths = sorted(img_dir.glob("*.png"))
    return paths


def get_already_processed(output_dir: Path) -> set[str]:
    """Get set of tweet IDs that already have OCR output."""
    if not output_dir.exists():
        return set()
    return {p.stem for p in output_dir.glob("*.json")}


# ── PaddleOCR engine ────────────────────────────────────────────────────────

class PaddleOCREngine:
    """Wrapper for PaddleOCR 3.x (PP-OCRv5)."""

    def __init__(self, device: str = "gpu:0"):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        self.name = "paddleocr_3.x_pp-ocrv5"

    def extract(self, image_path: str) -> dict:
        """Extract OCR text from an image.

        Returns dict with ocr_text, confidence, num_boxes, details.
        """
        # predict() is a generator in PaddleOCR 3.x — consume it
        result = list(self.ocr.predict(image_path))

        texts = []
        confidences = []

        for page_result in result:
            # PaddleOCR 3.x / PaddleX result objects use dict-style [] access
            # rec_texts and rec_scores are top-level keys, NOT nested under "res"
            try:
                rec_texts = list(page_result["rec_texts"])
                rec_scores = page_result["rec_scores"]
                if hasattr(rec_scores, "tolist"):
                    rec_scores = rec_scores.tolist()
                else:
                    rec_scores = list(rec_scores)
            except (KeyError, TypeError):
                rec_texts, rec_scores = [], []

            texts.extend(rec_texts)
            confidences.extend(rec_scores)

        combined = " ".join(str(t) for t in texts if str(t).strip())
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "ocr_text": combined,
            "confidence": round(avg_conf, 4),
            "num_boxes": len(texts),
            "details": [
                {"text": str(t), "score": round(float(s), 4)}
                for t, s in zip(texts, confidences)
            ],
        }


# ── EasyOCR engine (fallback) ───────────────────────────────────────────────

class EasyOCREngine:
    """Wrapper for EasyOCR as a fallback."""

    def __init__(self, device: str = "gpu:0"):
        import easyocr

        gpu = "gpu" in device.lower()
        self.reader = easyocr.Reader(["en"], gpu=gpu)
        self.name = "easyocr"

    def extract(self, image_path: str) -> dict:
        """Extract OCR text from an image."""
        results = self.reader.readtext(image_path)

        texts = [r[1] for r in results]
        confidences = [float(r[2]) for r in results]

        combined = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "ocr_text": combined,
            "confidence": round(avg_conf, 4),
            "num_boxes": len(texts),
            "details": [
                {"text": t, "score": round(s, 4)}
                for t, s in zip(texts, confidences)
            ],
        }


# ── Main processing ─────────────────────────────────────────────────────────

def process_images(
    engine,
    image_paths: list[Path],
    output_dir: Path,
    consolidated_path: Path,
    failure_log: Path,
    resume: bool = False,
) -> None:
    """Process all images and save OCR results."""
    from tqdm import tqdm

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing consolidated data if resuming
    consolidated = {}
    if resume and consolidated_path.exists():
        with open(consolidated_path, "r", encoding="utf-8") as f:
            consolidated = json.load(f)

    # Determine which images to process
    already_done = get_already_processed(output_dir) if resume else set()
    to_process = [p for p in image_paths if p.stem not in already_done]

    print(f"[INFO] Total images: {len(image_paths)}")
    print(f"[INFO] Already processed: {len(already_done)}")
    print(f"[INFO] To process: {len(to_process)}")
    print(f"[INFO] OCR engine: {engine.name}")

    failures = []
    start_time = time.time()

    for img_path in tqdm(to_process, desc="OCR extraction"):
        tweet_id = img_path.stem
        try:
            result = engine.extract(str(img_path))
            result["tweet_id"] = tweet_id
            result["ocr_engine"] = engine.name

            # Save per-image JSON
            out_file = output_dir / f"{tweet_id}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)

            # Update consolidated
            consolidated[tweet_id] = result["ocr_text"]

        except Exception as e:
            failures.append(f"{tweet_id}: {e}")

        # Periodic save of consolidated (every 1000 images)
        if len(consolidated) % 1000 == 0 and consolidated:
            with open(consolidated_path, "w", encoding="utf-8") as f:
                json.dump(consolidated, f, ensure_ascii=False)

    # Final save
    with open(consolidated_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, ensure_ascii=False)
    print(f"[INFO] Saved consolidated OCR → {consolidated_path} ({len(consolidated)} entries)")

    # Save failures
    if failures:
        with open(failure_log, "w") as f:
            f.write("\n".join(failures))
        print(f"[WARN] {len(failures)} failures logged → {failure_log}")

    elapsed = time.time() - start_time
    per_img = elapsed / max(len(to_process), 1)
    print(f"[INFO] Done in {elapsed:.0f}s ({per_img:.3f}s/image)")


def main():
    parser = argparse.ArgumentParser(description="Re-extract OCR for MMHS150K images")
    parser.add_argument("--img-dir", type=str, default=str(IMG_DIR), help="Input image directory")
    parser.add_argument("--output-dir", type=str, default=str(OCR_DIR_NEW), help="Output JSON directory")
    parser.add_argument("--device", type=str, default="gpu:0", help="Device: gpu:0, cpu, etc.")
    parser.add_argument("--engine", type=str, default="paddleocr", choices=["paddleocr", "easyocr"])
    parser.add_argument("--fresh", action="store_true", help="Ignore existing files and restart from scratch")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    output_dir = Path(args.output_dir)
    consolidated_path = Path(args.output_dir).parent / "ocr_consolidated.json"
    failure_log = Path(args.output_dir).parent / "ocr_failures.txt"

    # Auto-resume by default: skip images already processed
    resume = not args.fresh

    image_paths = get_image_paths(img_dir)
    if not image_paths:
        print(f"[ERROR] No images found in {img_dir}")
        sys.exit(1)

    # Initialize OCR engine
    if args.engine == "paddleocr":
        try:
            engine = PaddleOCREngine(device=args.device)
        except ImportError:
            print("[WARN] PaddleOCR not available, falling back to EasyOCR")
            engine = EasyOCREngine(device=args.device)
    else:
        engine = EasyOCREngine(device=args.device)

    process_images(
        engine=engine,
        image_paths=image_paths,
        output_dir=output_dir,
        consolidated_path=consolidated_path,
        failure_log=failure_log,
        resume=resume,
    )


if __name__ == "__main__":
    main()
