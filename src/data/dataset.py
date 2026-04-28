"""PyTorch Dataset class for MMHS150K multimodal hate speech detection.

Returns a dict per sample (not tuple) as required by coding standards.
Images are loaded lazily in __getitem__, not __init__.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.preprocessing import clean_ocr_text, clean_tweet_text
from src.data.splits import load_gt_json, load_ocr_data, load_processed_labels, load_split_ids
from src.data.transforms import get_base_transforms
from src.utils.config import IMG_DIR, DataConfig

logger = logging.getLogger(__name__)


def _load_image(path: Path) -> Optional[Image.Image]:
    """Load and validate an image. Returns RGB PIL Image or None on failure."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return None


class MMHS150KDataset(Dataset):
    """MMHS150K multimodal dataset for Track A (classical DL fusion).

    Each sample returns a dict:
        image: Tensor (C, H, W)
        tweet_text: str (preprocessed)
        ocr_text: str (preprocessed, may be "")
        label: int (hard 6-class majority vote)
        label_binary: int (0=NotHate, 1=Hate)
        soft_label: Tensor (6,)
        tweet_id: str
        agreement_level: int (3, 2, or 1)
    """

    def __init__(
        self,
        split: str,
        ocr_source: str = "old",
        transform=None,
        config: Optional[DataConfig] = None,
    ):
        """
        Args:
            split: "train", "val", or "test".
            ocr_source: "old" (original img_txt/), "new" (re-extracted), "both".
            transform: torchvision transform pipeline. Defaults to base transforms.
            config: DataConfig with preprocessing flags.
        """
        self.split = split
        self.config = config or DataConfig()
        self.transform = transform or get_base_transforms(self.config.img_size)
        self.img_dir = Path(self.config.img_dir)
        self.convert_emoji = self.config.convert_emoji

        # Cache metadata in __init__ (loaded ONCE)
        self.gt_data = load_gt_json()
        self.labels = load_processed_labels()
        self.split_ids = load_split_ids(split)
        self.ocr_data = load_ocr_data(ocr_source)

        # Filter out IDs that don't exist in labels (safety check)
        valid = [sid for sid in self.split_ids if sid in self.labels]
        if len(valid) < len(self.split_ids):
            logger.warning(
                f"Filtered {len(self.split_ids) - len(valid)} IDs missing from processed labels"
            )
        self.split_ids = valid

        # Optionally exclude full-disagreement samples
        if self.config.exclude_full_disagreement:
            before = len(self.split_ids)
            self.split_ids = [
                sid for sid in self.split_ids
                if self.labels[sid]["agreement_level"] > 1
            ]
            logger.info(
                f"Excluded {before - len(self.split_ids)} full-disagreement samples from {split}"
            )

        # Placeholder image: a black image tensor for corrupt/missing images
        self._placeholder_img = torch.zeros(3, self.config.img_size, self.config.img_size)

    def __len__(self) -> int:
        return len(self.split_ids)

    def __getitem__(self, idx: int) -> dict:
        tweet_id = self.split_ids[idx]
        entry = self.gt_data[tweet_id]
        label_info = self.labels[tweet_id]

        # Load image LAZILY — in __getitem__, not __init__
        img_path = self.img_dir / f"{tweet_id}.jpg"
        pil_img = _load_image(img_path)
        if pil_img is not None:
            image = self.transform(pil_img)
        else:
            image = self._placeholder_img.clone()

        # Get and clean texts
        tweet_text = clean_tweet_text(entry["tweet_text"], convert_emoji=self.convert_emoji)
        ocr_text = clean_ocr_text(self.ocr_data.get(tweet_id, ""))

        return {
            "image": image,
            "tweet_text": tweet_text,
            "ocr_text": ocr_text,
            "label": label_info["hard_label_6class"],
            "label_binary": label_info["hard_label_binary"],
            "soft_label": torch.tensor(label_info["soft_label_6class"], dtype=torch.float32),
            "tweet_id": tweet_id,
            "agreement_level": label_info["agreement_level"],
        }


def mmhs_collate_fn(batch: list[dict]) -> dict:
    """Custom collate function for MMHS150KDataset.

    Stacks tensors and collects strings into lists since the default
    collate can't handle mixed tensor/string batches.
    """
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "tweet_text": [b["tweet_text"] for b in batch],
        "ocr_text": [b["ocr_text"] for b in batch],
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "label_binary": torch.tensor([b["label_binary"] for b in batch], dtype=torch.long),
        "soft_label": torch.stack([b["soft_label"] for b in batch]),
        "tweet_id": [b["tweet_id"] for b in batch],
        "agreement_level": torch.tensor([b["agreement_level"] for b in batch], dtype=torch.long),
    }
