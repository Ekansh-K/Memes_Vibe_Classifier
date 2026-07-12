"""Shared Stage-1 dataset: tweet + OCR + caption + soft binary labels."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import clean_ocr_text, clean_tweet_text
from src.data.splits import load_gt_json, load_ocr_data, load_processed_labels, load_split_ids
from src.utils.config import IMG_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

TextMode = Literal["tweet", "tweet_ocr", "all_text"]
VLM_CAPTIONS_FILE = PROJECT_ROOT / "results" / "vlm_captions.json"

_captions_cache: Optional[dict] = None


def load_vlm_captions() -> dict:
    global _captions_cache
    if _captions_cache is not None:
        return _captions_cache
    # Prefer results/, fall back to dataset/
    candidates = [
        VLM_CAPTIONS_FILE,
        PROJECT_ROOT / "dataset" / "vlm_captions.json",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        logger.warning("[S1] No VLM captions found; all_text falls back to tweet+OCR.")
        _captions_cache = {}
        return _captions_cache
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # Support {id: {caption: ...}} or {id: str}
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out[k] = v.get("caption", "") or ""
        else:
            out[k] = str(v) if v else ""
    _captions_cache = out
    logger.info(f"[S1] Loaded {len(_captions_cache):,} VLM captions from {path}")
    return _captions_cache


def build_text(
    tweet_id: str,
    tweet_text: str,
    ocr_text: str,
    captions: dict,
    text_mode: TextMode,
) -> str:
    caption = captions.get(tweet_id, "")
    if text_mode == "tweet":
        return tweet_text or ""
    if text_mode == "tweet_ocr":
        parts = [p for p in (tweet_text, ocr_text) if p]
        return " [SEP] ".join(parts)
    # all_text
    parts = [p for p in (caption, ocr_text, tweet_text) if p]
    if not parts:
        parts = [p for p in (tweet_text, ocr_text) if p]
    return " [SEP] ".join(parts)


def soft_hate_prob(label_info: dict) -> float:
    """P(hate) from soft_label_binary [p_nothate, p_hate] or hard fallback."""
    sb = label_info.get("soft_label_binary")
    if sb is not None and len(sb) >= 2:
        return float(sb[1])
    return float(label_info["hard_label_binary"])


class Stage1Dataset(Dataset):
    """Binary hate Stage-1 samples with soft targets + agreement."""

    def __init__(
        self,
        split: str,
        text_mode: TextMode = "all_text",
        ocr_source: str = "filtered",
        load_images: bool = False,
        img_size: int = 224,
        max_samples: Optional[int] = None,
        exclude_full_disagreement: bool = False,
        min_agreement_binary: Optional[int] = None,
        seed: int = 42,
        use_image_store: bool = False,
    ):
        self.split = split
        self.text_mode = text_mode
        self.load_images = load_images
        self.img_size = img_size
        self.img_dir = Path(IMG_DIR)
        self.seed = seed
        self._min_agreement_binary = min_agreement_binary

        self.gt_data = load_gt_json()
        self.labels = load_processed_labels()
        self.ocr_data = load_ocr_data(ocr_source)
        self.captions = load_vlm_captions()
        if text_mode == "all_text" and not self.captions:
            logger.warning(
                "[S1] text_mode=all_text but no VLM captions found "
                f"(checked results/vlm_captions.json and dataset/vlm_captions.json). "
                "Falling back to tweet+OCR. Copy captions to the remote machine "
                "or use --text_mode tweet_ocr."
            )

        all_ids = load_split_ids(split)
        valid_ids = [sid for sid in all_ids if sid in self.labels]
        if exclude_full_disagreement:
            valid_ids = [
                sid
                for sid in valid_ids
                if int(
                    self.labels[sid].get(
                        "agreement_binary", self.labels[sid]["agreement_level"]
                    )
                )
                > 1
            ]
        # Optional: keep only samples with binary agreement >= threshold (soft recipe S6)
        min_agr = getattr(self, "_min_agreement_binary", None)
        if min_agr is not None:
            valid_ids = [
                sid
                for sid in valid_ids
                if int(
                    self.labels[sid].get(
                        "agreement_binary", self.labels[sid]["agreement_level"]
                    )
                )
                >= min_agr
            ]
        if max_samples is not None and max_samples < len(valid_ids):
            valid_ids = self._stratified_sample(valid_ids, max_samples)

        self.sample_ids = valid_ids
        self._img_store = None
        if use_image_store and load_images:
            try:
                from src.data.image_store import load_image_store
                self._img_store = load_image_store(img_size)
            except Exception as e:
                logger.warning(f"[S1] Image store unavailable: {e}")

        logger.info(
            f"[S1] {split}: {len(self.sample_ids):,} samples  "
            f"text_mode={text_mode}  images={load_images}"
        )

    def _stratified_sample(self, ids: list, n: int) -> list:
        hate = [s for s in ids if self.labels[s]["hard_label_binary"] == 1]
        nothate = [s for s in ids if self.labels[s]["hard_label_binary"] == 0]
        ratio = len(hate) / max(len(ids), 1)
        n_hate = max(1, round(n * ratio))
        n_nothate = n - n_hate
        rng = random.Random(self.seed)
        sampled = (
            rng.sample(hate, min(n_hate, len(hate)))
            + rng.sample(nothate, min(n_nothate, len(nothate)))
        )
        rng.shuffle(sampled)
        return sampled

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image(self, tweet_id: str) -> Image.Image:
        if self._img_store is not None and tweet_id in self._img_store:
            arr = self._img_store[tweet_id]
            if arr is not None:
                return Image.fromarray(arr)
        path = self.img_dir / f"{tweet_id}.jpg"
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception:
            return Image.new("RGB", (self.img_size, self.img_size), (0, 0, 0))

    def __getitem__(self, idx: int) -> dict:
        tweet_id = self.sample_ids[idx]
        entry = self.gt_data[tweet_id]
        info = self.labels[tweet_id]

        tweet_text = clean_tweet_text(entry["tweet_text"], convert_emoji=False)
        ocr_text = clean_ocr_text(self.ocr_data.get(tweet_id, ""))
        text = build_text(tweet_id, tweet_text, ocr_text, self.captions, self.text_mode)

        # Short CLIP-friendly string (OCR priority then tweet; 77-token limit later)
        clip_text = " ".join(p for p in (ocr_text, tweet_text) if p) or text

        agreement_fine = int(info["agreement_level"])
        agreement_bin = int(info.get("agreement_binary", agreement_fine))
        item = {
            "tweet_id": tweet_id,
            "text": text,
            "clip_text": clip_text[:400],
            "label_binary": int(info["hard_label_binary"]),
            "soft_hate": soft_hate_prob(info),
            "agreement_level": agreement_fine,
            "agreement_binary": agreement_bin,
        }
        if self.load_images:
            item["image"] = self._load_image(tweet_id)
        return item


def stage1_collate_text(batch: list[dict]) -> dict:
    return {
        "tweet_id": [b["tweet_id"] for b in batch],
        "text": [b["text"] for b in batch],
        "label_binary": torch.tensor([b["label_binary"] for b in batch], dtype=torch.long),
        "soft_hate": torch.tensor([b["soft_hate"] for b in batch], dtype=torch.float32),
        "agreement_level": torch.tensor(
            [b["agreement_level"] for b in batch], dtype=torch.long
        ),
        "agreement_binary": torch.tensor(
            [b.get("agreement_binary", b["agreement_level"]) for b in batch],
            dtype=torch.long,
        ),
    }


def stage1_collate_multimodal(batch: list[dict]) -> dict:
    out = stage1_collate_text(batch)
    out["image"] = [b["image"] for b in batch]
    out["clip_text"] = [b["clip_text"] for b in batch]
    return out
