"""P7-specific dataset and collate function.

Extends the base MMHS150KDataset with:
- Stage filtering (stage=2 keeps only hateful samples for category classification)
- text_mode control (no_caption / tweet_ocr / all_text caption ablation)
- VLM caption loading from results/vlm_captions.json
- multi_label_binary field (6-element float tensor, derived from soft_label_6class)
- BERT tokenization of the constructed text string

Returns a dict per sample with all fields needed by MHSDF.
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

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

from src.data.preprocessing import clean_ocr_text, clean_tweet_text
from src.data.splits import load_gt_json, load_ocr_data, load_processed_labels, load_split_ids
from src.data.transforms import get_base_transforms, get_train_transforms
from src.data.image_store import load_image_store
from src.p7.config import P7Config
from src.p7.tokenizer import P7Tokenizer
from src.utils.config import PROJECT_ROOT, IMG_DIR

logger = logging.getLogger(__name__)

# ── Hate category label map for Stage 2 (NotHate excluded) ───────────────────
# Maps original 6-class label → Stage 2 category index (0-4)
# 0=NotHate is excluded; 1-5 → 0-4
HATE_CATEGORIES = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
HATE_CAT_NAMES = ["Racist", "Sexist", "Homophobe", "Religion", "OtherHate"]

# ── Caption file path ─────────────────────────────────────────────────────────
VLM_CAPTIONS_FILE = PROJECT_ROOT / "results" / "vlm_captions.json"

# ── Module-level cache for captions ─────────────────────────────────────────
_captions_cache: Optional[dict] = None


def load_vlm_captions() -> dict:
    """Load VLM captions from results/vlm_captions.json.

    Format: {tweet_id: {"caption": str, "elapsed_s": float}}
    Returns: {tweet_id: caption_str}  — empty dict if file not found.
    """
    global _captions_cache
    if _captions_cache is not None:
        return _captions_cache

    if not VLM_CAPTIONS_FILE.exists():
        logger.warning(
            f"[P7] VLM captions not found at {VLM_CAPTIONS_FILE}. "
            "Text mode 'all_text' will fall back to tweet+OCR only."
        )
        _captions_cache = {}
        return _captions_cache

    with open(VLM_CAPTIONS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    _captions_cache = {k: v["caption"] for k, v in raw.items()}
    logger.info(f"[P7] Loaded {len(_captions_cache):,} VLM captions.")
    return _captions_cache


# ── Token cache ───────────────────────────────────────────────────────────────
_token_cache: dict[str, dict] = {}   # key = "<text_mode>" → {tweet_id: np.array}


def _token_cache_path(text_mode: str, max_seq_len: int) -> Path:
    return PROJECT_ROOT / "dataset" / f"token_cache_{text_mode}_{max_seq_len}.npy"


def build_token_cache(
    text_mode: str,
    max_seq_len: int,
    tokenizer: "P7Tokenizer",
    force: bool = False,
) -> dict:
    """Pre-tokenize ALL samples and save to a .npy file for fast lookup.

    Only needs to run once per (text_mode, max_seq_len) combination.
    Subsequent calls just load from disk.

    Returns:
        Dict mapping tweet_id → np.ndarray of shape (max_seq_len,) int32.
    """
    cache_key = f"{text_mode}_{max_seq_len}"
    if cache_key in _token_cache and not force:
        return _token_cache[cache_key]

    cache_file = _token_cache_path(text_mode, max_seq_len)
    if cache_file.exists() and not force:
        logger.info(f"[P7] Loading token cache from {cache_file} ...")
        data = np.load(cache_file, allow_pickle=True).item()
        _token_cache[cache_key] = data
        logger.info(f"[P7] Token cache loaded: {len(data):,} samples.")
        return data

    logger.info(f"[P7] Building token cache for text_mode='{text_mode}' ...")
    gt_data   = load_gt_json()
    ocr_data  = load_ocr_data("new")
    captions  = load_vlm_captions()

    result = {}
    for i, (tweet_id, entry) in enumerate(gt_data.items()):
        tweet_text = clean_tweet_text(entry["tweet_text"], convert_emoji=False)
        ocr_text   = clean_ocr_text(ocr_data.get(tweet_id, ""))
        caption    = captions.get(tweet_id, "")

        if text_mode in ("no_caption", "tweet_ocr"):
            text = f"{tweet_text} {ocr_text}".strip()
        else:  # all_text
            parts = [p for p in [caption, ocr_text, tweet_text] if p]
            text  = " ".join(parts).strip()

        ids = tokenizer.encode(text)
        result[tweet_id] = np.array(ids, dtype=np.int32)

        if (i + 1) % 10_000 == 0:
            logger.info(f"[P7] Token cache: {i+1:,}/{len(gt_data):,} done ...")

    np.save(cache_file, result)
    _token_cache[cache_key] = result
    logger.info(f"[P7] Token cache saved → {cache_file}  ({len(result):,} samples)")
    return result


def compute_multilabel_from_soft(
    soft_label: list[float], threshold: float = 2 / 3
) -> list[float]:
    """Derive a 6-element multi-hot vector from soft (probability) labels.

    Class c is active (1.0) if soft_label[c] >= threshold (i.e., >= 2 of 3
    annotators voted for it).

    Args:
        soft_label: 6-element list of vote fractions from processed_labels.json.
        threshold:  Minimum fraction to consider a class active (default 2/3).

    Returns:
        List of 6 floats (0.0 or 1.0) for BCEWithLogitsLoss compatibility.
    """
    return [1.0 if s >= threshold else 0.0 for s in soft_label]


def _load_image(path: Path, placeholder: torch.Tensor) -> torch.Tensor:
    """Load a PIL image and apply transforms. Returns placeholder on failure."""
    try:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.debug(f"[P7] Failed to load image {path}: {e}")
        return None


class P7Dataset(Dataset):
    """P7-specific dataset extending MMHS150K with stage filtering and text modes.

    Each sample returns a dict:
        image:              Tensor (3, H, W)
        token_ids:          LongTensor (max_seq_len,)  — BERT-tokenized text
        text:               str  — constructed text string
        tweet_id:           str
        label_binary:       int  (0=NotHate, 1=Hate)       — Stage 1 target
        label_6class:       int  (0-5 majority vote)        — P7-B target
        label_s2:           int  (0-4 hate category, -1 if NotHate) — P7-C Stage 2 target
        multi_label_binary: FloatTensor (6,)               — P7-D Stage 2 target
        soft_label:         FloatTensor (6,)
        agreement_level:    int
    """

    def __init__(
        self,
        split: str,
        config: P7Config,
        tokenizer: P7Tokenizer,
        stage: Optional[int] = None,
        is_training: bool = False,
        max_samples: Optional[int] = None,
        token_cache: Optional[dict] = None,
    ):
        """
        Args:
            split:       "train", "val", or "test".
            config:      P7Config instance.
            tokenizer:   P7Tokenizer instance (shared across train/val/test).
            stage:       1 = full dataset (binary labels), 2 = hateful-only subset.
                         None = use all samples (for P7-A/B direct modes).
            is_training: If True, applies training augmentation transforms.
            max_samples: If set, stratified-sample this many samples (preserves
                         class ratio). Useful for fast smoke tests.
            token_cache: Pre-computed token ID dict {tweet_id → np.ndarray}.
                         If provided, skips the HuggingFace tokenizer in __getitem__.
        """
        self.split = split
        self.config = config
        self.tokenizer = tokenizer
        self.stage = stage
        self.text_mode = config.text_mode
        self.token_cache = token_cache
        self.is_training = is_training
        self.token_drop_rate = getattr(config, "token_drop_rate", 0.0) if is_training else 0.0

        # Image transforms
        _use_erasing = getattr(config, "use_random_erasing", False)
        if is_training:
            self.transform = get_train_transforms(config.img_size, use_random_erasing=_use_erasing)
        else:
            self.transform = get_base_transforms(config.img_size)

        self.img_dir = Path(IMG_DIR)
        self._placeholder_img = torch.zeros(3, config.img_size, config.img_size)

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
                f"[P7] {split}: Filtered {len(all_ids) - len(valid_ids)} IDs "
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
                f"[P7] {split}: Excluded {before - len(valid_ids)} full-disagreement samples."
            )

        # Stage 2: keep only hateful samples (label_binary == 1)
        if stage == 2:
            before = len(valid_ids)
            valid_ids = [
                sid for sid in valid_ids
                if self.labels[sid]["hard_label_binary"] == 1
            ]
            logger.info(
                f"[P7] {split} Stage 2: Kept {len(valid_ids)}/{before} hateful-only samples."
            )

        # ── Stratified subset sampling ────────────────────────────────────
        if max_samples is not None and max_samples < len(valid_ids):
            valid_ids = self._stratified_sample(valid_ids, max_samples)
            logger.info(
                f"[P7] {split}: Stratified subset → {len(valid_ids):,} samples "
                f"(max_samples={max_samples})"
            )

        self.sample_ids = valid_ids
        logger.info(f"[P7] {split} dataset: {len(self.sample_ids):,} samples  (stage={stage})")

        # ── Image store (memmap, fast JPEG-free loading) ───────────────────────────
        if getattr(config, "use_image_store", False):
            self._img_store = load_image_store(config.img_size)
            if self._img_store is None:
                logger.warning(
                    "[P7] ImageStore not built yet. Run build_image_store() first. "
                    "Falling back to disk loading."
                )
        else:
            self._img_store = None

        # ── Legacy RAM image cache (dict of numpy arrays) ────────────────────────
        self._image_cache: dict[str, np.ndarray] = {}
        if getattr(config, "use_image_cache", False):
            self._build_image_cache()

    def _stratified_sample(self, ids: list, n: int) -> list:
        """Stratified sample n IDs preserving binary class ratio."""
        hate    = [sid for sid in ids if self.labels[sid]["hard_label_binary"] == 1]
        nothate = [sid for sid in ids if self.labels[sid]["hard_label_binary"] == 0]
        ratio   = len(hate) / len(ids)
        n_hate    = max(1, round(n * ratio))
        n_nothate = n - n_hate
        rng = random.Random(self.config.seed)
        sampled = rng.sample(hate, min(n_hate, len(hate))) + \
                  rng.sample(nothate, min(n_nothate, len(nothate)))
        rng.shuffle(sampled)
        return sampled

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _build_image_cache(self) -> None:
        """Pre-load all images into RAM as uint8 numpy arrays.

        Stores HxWxC uint8 arrays (not float32 tensors) to minimise RAM.
        Transforms (augmentation, normalisation) are still applied lazily
        in __getitem__, so training augmentation keeps working correctly.

        Safety: estimates required RAM and skips if < 70% of free RAM is
        available, so we never OOM the system.
        """
        n = len(self.sample_ids)
        img_size = self.config.img_size
        est_bytes = n * img_size * img_size * 3  # uint8, 3 channels

        if _PSUTIL_AVAILABLE:
            available = psutil.virtual_memory().available
            if est_bytes > available * 0.70:
                logger.warning(
                    f"[P7] Image cache SKIPPED: need {est_bytes/1e9:.1f}GB, "
                    f"only {available/1e9:.1f}GB available (threshold 70%). "
                    "Falling back to disk loading."
                )
                return
            logger.info(
                f"[P7] Caching {n:,} images in RAM "
                f"(~{est_bytes/1e9:.1f}GB / {available/1e9:.1f}GB free) ..."
            )
        else:
            logger.info(
                f"[P7] Caching {n:,} images in RAM "
                f"(~{est_bytes/1e9:.1f}GB, psutil not installed — no RAM check) ..."
            )

        loaded = 0
        for i, tweet_id in enumerate(self.sample_ids):
            img_path = self.img_dir / f"{tweet_id}.jpg"
            try:
                with Image.open(img_path) as im:
                    self._image_cache[tweet_id] = np.array(
                        im.convert("RGB"), dtype=np.uint8
                    )
                loaded += 1
            except Exception:
                pass  # missing images fall back to placeholder in __getitem__

            if (i + 1) % 5_000 == 0:
                logger.info(f"[P7] Image cache: {i+1:,}/{n:,} loaded ...")

        logger.info(f"[P7] Image cache ready: {loaded:,}/{n:,} images in RAM.")

    def _build_text(self, tweet_id: str, tweet_text: str, ocr_text: str) -> str:
        """Construct the text input string for the BiLSTM based on text_mode."""
        caption = self.captions.get(tweet_id, "")

        if self.text_mode == "no_caption":
            return f"{tweet_text} {ocr_text}".strip()
        elif self.text_mode == "tweet_ocr":
            return f"{tweet_text} {ocr_text}".strip()
        else:  # all_text
            parts = [p for p in [caption, ocr_text, tweet_text] if p]
            return " ".join(parts).strip()

    def __getitem__(self, idx: int) -> dict:
        tweet_id = self.sample_ids[idx]
        entry = self.gt_data[tweet_id]
        label_info = self.labels[tweet_id]

        # ── Image ─────────────────────────────────────────────────────────────
        img_path = self.img_dir / f"{tweet_id}.jpg"
        if self._img_store is not None and tweet_id in self._img_store:
            # ⚡ Fast path: memmap read (no JPEG decode, no disk seek)
            pil_img = Image.fromarray(self._img_store[tweet_id])
            image = self.transform(pil_img)
        elif tweet_id in self._image_cache:
            # ⚡ Legacy RAM cache fast path
            pil_img = Image.fromarray(self._image_cache[tweet_id])
            image = self.transform(pil_img)
        else:
            # 🐢 Slow path: JPEG decode from disk
            pil_img = _load_image(img_path, self._placeholder_img)
            if pil_img is not None:
                image = self.transform(pil_img)
            else:
                image = self._placeholder_img.clone()

        # ── Text ─────────────────────────────────────────────────────────────
        tweet_text = clean_tweet_text(
            entry["tweet_text"], convert_emoji=False
        )
        ocr_text = clean_ocr_text(self.ocr_data.get(tweet_id, ""))
        text = self._build_text(tweet_id, tweet_text, ocr_text)

        # Fast path: use pre-built token cache if available
        if self.token_cache is not None and tweet_id in self.token_cache:
            token_ids = torch.tensor(self.token_cache[tweet_id], dtype=torch.long)
        else:
            token_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        # Random token drop (training only) — zeroes random interior tokens
        # Keeps first & last positions (CLS/SEP equivalents).
        if self.token_drop_rate > 0.0 and len(token_ids) > 2:
            mask = torch.ones(len(token_ids), dtype=torch.bool)
            mask[1:-1] = torch.bernoulli(
                torch.full((len(token_ids) - 2,), 1.0 - self.token_drop_rate)
            ).bool()
            token_ids = torch.where(mask, token_ids, torch.zeros_like(token_ids))

        # ── Labels ───────────────────────────────────────────────────────────
        hard_6 = label_info["hard_label_6class"]
        label_binary = label_info["hard_label_binary"]
        soft = label_info["soft_label_6class"]

        # Stage 2 category: remap hate classes 1-5 → 0-4; NotHate → -1
        label_s2 = HATE_CATEGORIES.get(hard_6, -1)

        # Multi-label binary vector (6-element float, for P7-D)
        multi_label = compute_multilabel_from_soft(
            soft, threshold=self.config.multilabel_threshold
        )

        return {
            "image": image,
            "token_ids": token_ids,
            "text": text,
            "tweet_id": tweet_id,
            "label_binary": label_binary,
            "label_6class": hard_6,
            "label_s2": label_s2,
            "multi_label_binary": torch.tensor(multi_label, dtype=torch.float32),
            "soft_label": torch.tensor(soft, dtype=torch.float32),
            "agreement_level": label_info["agreement_level"],
        }


def p7_collate_fn(batch: list[dict]) -> dict:
    """Collate function for P7Dataset.

    Stacks tensors, collects strings into lists.
    """
    return {
        "image":              torch.stack([b["image"] for b in batch]),
        "token_ids":          torch.stack([b["token_ids"] for b in batch]),
        "text":               [b["text"] for b in batch],
        "tweet_id":           [b["tweet_id"] for b in batch],
        "label_binary":       torch.tensor([b["label_binary"] for b in batch], dtype=torch.long),
        "label_6class":       torch.tensor([b["label_6class"] for b in batch], dtype=torch.long),
        "label_s2":           torch.tensor([b["label_s2"] for b in batch], dtype=torch.long),
        "multi_label_binary": torch.stack([b["multi_label_binary"] for b in batch]),
        "soft_label":         torch.stack([b["soft_label"] for b in batch]),
        "agreement_level":    torch.tensor([b["agreement_level"] for b in batch], dtype=torch.long),
    }
