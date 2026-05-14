"""P2-specific dataset and collate function.

Extends the base MMHS150K data infrastructure with:
- Stage filtering (stage=2 keeps only hateful samples)
- text_mode control (tweet_ocr / all_text)
- VLM caption loading from results/vlm_captions.json
- Returns PIL images (CLIP preprocess applied in model forward)
- Returns raw text strings (TweetEval tokenization in model forward)
- multi_label_binary field for P2-D
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import clean_ocr_text, clean_tweet_text
from src.data.splits import load_gt_json, load_ocr_data, load_processed_labels, load_split_ids
from src.data.transforms import get_base_transforms
from src.p2.config import P2Config
from src.utils.config import PROJECT_ROOT, IMG_DIR

logger = logging.getLogger(__name__)

# ── Hate category label map for Stage 2 (NotHate excluded) ───────────────────
HATE_CATEGORIES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
HATE_CAT_NAMES = ["Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]

# ── Caption file path ─────────────────────────────────────────────────────────
VLM_CAPTIONS_FILE = PROJECT_ROOT / "results" / "vlm_captions.json"

# ── Module-level cache for captions ─────────────────────────────────────────
_captions_cache: Optional[dict] = None


def load_vlm_captions() -> dict:
    """Load VLM captions from results/vlm_captions.json.

    Format: {tweet_id: {"caption": str, "elapsed_s": float}}
    Returns: {tweet_id: caption_str} — empty dict if file not found.
    """
    global _captions_cache
    if _captions_cache is not None:
        return _captions_cache

    if not VLM_CAPTIONS_FILE.exists():
        logger.warning(
            f"[P2] VLM captions not found at {VLM_CAPTIONS_FILE}. "
            "Text mode 'all_text' will fall back to tweet+OCR only."
        )
        _captions_cache = {}
        return _captions_cache

    with open(VLM_CAPTIONS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _captions_cache = {k: v["caption"] for k, v in raw.items()}
    logger.info(f"[P2] Loaded {len(_captions_cache):,} VLM captions.")
    return _captions_cache


def compute_multilabel_from_soft(
    soft_label: list[float], threshold: float = 2 / 3
) -> list[float]:
    """Derive a 6-element multi-hot vector from soft (probability) labels.

    Class c is active (1.0) if soft_label[c] >= threshold (i.e., >= 2 of 3
    annotators voted for it).
    """
    return [1.0 if s >= threshold else 0.0 for s in soft_label]


def _load_pil_image(path: Path) -> Optional[Image.Image]:
    """Load an image as RGB PIL Image. Returns None on failure."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.debug(f"[P2] Failed to load image {path}: {e}")
        return None


class P2Dataset(Dataset):
    """P2-specific dataset for TCAM pipeline.

    Each sample returns a dict:
        image:              PIL.Image (RGB) — CLIP preprocess applied in collate/model
        text:               str — constructed from text_mode
        tweet_id:           str
        label_binary:       int (0=NotHate, 1=Hate)
        label_6class:       int (0-5 majority vote)
        label_s2:           int (0-4 hate category, -1 if NotHate)
        multi_label_binary: FloatTensor(6,)
        soft_label:         FloatTensor(6,)
        agreement_level:    int
    """

    def __init__(
        self,
        split: str,
        config: P2Config,
        stage: Optional[int] = None,
        is_training: bool = False,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split:       "train", "val", or "test".
            config:      P2Config instance.
            stage:       1 = full dataset (binary), 2 = hateful-only subset.
                         None = use all samples (for P2-A/B direct modes).
            is_training: If True, applies training augmentation transforms.
            max_samples: If set, stratified-sample this many samples.
        """
        self.split = split
        self.config = config
        self.stage = stage
        self.text_mode = config.text_mode
        self.is_training = is_training
        self.img_dir = Path(IMG_DIR)

        # Load metadata (cached at module level after first call)
        self.gt_data = load_gt_json()
        self.labels = load_processed_labels()
        self.ocr_data = load_ocr_data(config.ocr_source)
        self.captions = load_vlm_captions()

        # Build sample ID list
        all_ids = load_split_ids(split)

        # Filter to IDs present in processed labels
        valid_ids = [sid for sid in all_ids if sid in self.labels]
        if len(valid_ids) < len(all_ids):
            logger.warning(
                f"[P2] {split}: Filtered {len(all_ids) - len(valid_ids)} IDs "
                "missing from processed_labels.json"
            )

        # Optional: drop full-disagreement samples
        if config.exclude_full_disagreement:
            before = len(valid_ids)
            valid_ids = [
                sid for sid in valid_ids
                if self.labels[sid]["agreement_level"] > 1
            ]
            logger.info(
                f"[P2] {split}: Excluded {before - len(valid_ids)} "
                "full-disagreement samples."
            )

        # Stage 2: keep only hateful samples (label_binary == 1)
        if stage == 2:
            before = len(valid_ids)
            valid_ids = [
                sid for sid in valid_ids
                if self.labels[sid]["hard_label_binary"] == 1
            ]
            logger.info(
                f"[P2] {split} Stage 2: Kept {len(valid_ids)}/{before} "
                "hateful-only samples."
            )

        # Stratified subset sampling
        if max_samples is not None and max_samples < len(valid_ids):
            valid_ids = self._stratified_sample(valid_ids, max_samples)
            logger.info(
                f"[P2] {split}: Stratified subset → {len(valid_ids):,} samples "
                f"(max_samples={max_samples})"
            )

        self.sample_ids = valid_ids
        logger.info(
            f"[P2] {split} dataset: {len(self.sample_ids):,} samples "
            f"(stage={stage})"
        )

        # Image store (memmap, fast loading)
        self._img_store = None
        if getattr(config, "use_image_store", False):
            try:
                from src.data.image_store import load_image_store
                self._img_store = load_image_store(config.img_size)
                if self._img_store is None:
                    logger.warning(
                        "[P2] ImageStore not built. Falling back to disk loading."
                    )
            except ImportError:
                pass

    def _stratified_sample(self, ids: list, n: int) -> list:
        """Stratified sample n IDs preserving binary class ratio."""
        hate = [sid for sid in ids if self.labels[sid]["hard_label_binary"] == 1]
        nothate = [sid for sid in ids if self.labels[sid]["hard_label_binary"] == 0]
        ratio = len(hate) / len(ids)
        n_hate = max(1, round(n * ratio))
        n_nothate = n - n_hate
        rng = random.Random(self.config.seed)
        sampled = rng.sample(hate, min(n_hate, len(hate))) + \
                  rng.sample(nothate, min(n_nothate, len(nothate)))
        rng.shuffle(sampled)
        return sampled

    def _build_text(self, tweet_id: str, tweet_text: str, ocr_text: str) -> str:
        """Construct the text input string based on text_mode."""
        caption = self.captions.get(tweet_id, "")

        if self.text_mode == "tweet_ocr":
            parts = [p for p in [tweet_text, ocr_text] if p]
            return " [SEP] ".join(parts) if parts else ""
        else:  # all_text
            if caption:
                parts = [p for p in [caption, ocr_text, tweet_text] if p]
            else:
                # No caption available — fall back to tweet + ocr
                parts = [p for p in [tweet_text, ocr_text] if p]
            return " [SEP] ".join(parts) if parts else ""

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        tweet_id = self.sample_ids[idx]
        entry = self.gt_data[tweet_id]
        label_info = self.labels[tweet_id]

        # ── Image ─────────────────────────────────────────────────────────────
        pil_img = None
        if self._img_store is not None and tweet_id in self._img_store:
            arr = self._img_store[tweet_id]
            if arr is not None:
                pil_img = Image.fromarray(arr)
        if pil_img is None:
            img_path = self.img_dir / f"{tweet_id}.jpg"
            pil_img = _load_pil_image(img_path)
        if pil_img is None:
            # Black placeholder
            pil_img = Image.new("RGB", (self.config.img_size, self.config.img_size), (0, 0, 0))

        # ── Text ─────────────────────────────────────────────────────────────
        tweet_text = clean_tweet_text(entry["tweet_text"], convert_emoji=False)
        ocr_text = clean_ocr_text(self.ocr_data.get(tweet_id, ""))
        text = self._build_text(tweet_id, tweet_text, ocr_text)

        # ── Labels ───────────────────────────────────────────────────────────
        hard_6 = label_info["hard_label_6class"]
        label_binary = label_info["hard_label_binary"]
        soft = label_info["soft_label_6class"]

        # Stage 2 category: remap hate classes 1-5 → 0-4; NotHate → -1
        label_s2 = HATE_CATEGORIES.get(hard_6, -1)

        # Multi-label binary vector (6-element float, for P2-D)
        multi_label = compute_multilabel_from_soft(
            soft, threshold=self.config.multilabel_threshold
        )

        return {
            "image": pil_img,
            "text": text,
            "tweet_id": tweet_id,
            "label_binary": label_binary,
            "label_6class": hard_6,
            "label_s2": label_s2,
            "multi_label_binary": torch.tensor(multi_label, dtype=torch.float32),
            "soft_label": torch.tensor(soft, dtype=torch.float32),
            "agreement_level": label_info["agreement_level"],
        }


def p2_collate_fn(batch: list[dict]) -> dict:
    """Collate function for P2Dataset.

    Images stay as a list of PIL images — CLIP preprocess is applied
    in the model forward pass (on GPU).
    Texts stay as a list of strings — TweetEval tokenizes on the fly.
    Labels are stacked into tensors.
    """
    return {
        "image":              [b["image"] for b in batch],
        "text":               [b["text"] for b in batch],
        "tweet_id":           [b["tweet_id"] for b in batch],
        "label_binary":       torch.tensor([b["label_binary"] for b in batch], dtype=torch.long),
        "label_6class":       torch.tensor([b["label_6class"] for b in batch], dtype=torch.long),
        "label_s2":           torch.tensor([b["label_s2"] for b in batch], dtype=torch.long),
        "multi_label_binary": torch.stack([b["multi_label_binary"] for b in batch]),
        "soft_label":         torch.stack([b["soft_label"] for b in batch]),
        "agreement_level":    torch.tensor([b["agreement_level"] for b in batch], dtype=torch.long),
    }
