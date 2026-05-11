"""Shared numpy memmap image store for MMHS150K.

Eliminates per-sample JPEG decompression (the true bottleneck in __getitem__)
by pre-decoding all images once into a flat binary file and indexing them with
a tiny JSON lookup table.

Key properties:
    - Built once (~5–10 min), reused forever across all pipelines (P7, P3, etc.)
    - Full dataset (~150k images): ~22GB on disk; 30k subset: ~4.3GB
    - Memory: numpy memmap — OS page cache handles it; only accessed pages
      are in RAM, so actual RAM use ≈ working set (not 22GB)
    - num_workers safe: memmap is picklable (stores file path, not data).
      Each worker re-opens the file; no data is piped between processes.
    - Transforms (augmentation, normalization) still applied lazily per sample,
      so training augmentation keeps working correctly.

Usage:
    # Build once (run from a notebook cell or script):
    from src.data.image_store import build_image_store
    from src.utils.config import IMG_DIR
    store = build_image_store(Path(IMG_DIR))

    # Load in any dataset:
    from src.data.image_store import load_image_store
    store = load_image_store(img_size=224)   # None if not built yet
    if store and tweet_id in store:
        arr = store[tweet_id]          # np.ndarray (H, W, 3) uint8
        pil_img = Image.fromarray(arr)
        image = transform(pil_img)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.utils.config import PROJECT_ROOT, IMG_DIR

logger = logging.getLogger(__name__)

_STORE_DIR = PROJECT_ROOT / "dataset"

# Module-level cache: one ImageStore per img_size, per process
_store_cache: dict[int, "ImageStore"] = {}


def _bin_path(img_size: int) -> Path:
    return _STORE_DIR / f"image_store_{img_size}.bin"


def _idx_path(img_size: int) -> Path:
    return _STORE_DIR / f"image_store_{img_size}_index.json"


class ImageStore:
    """Read-only view into the pre-decoded memmap image store.

    Attributes:
        n_images (int):  Number of images in the store.
        img_size (int):  Pixel side length (images are square).

    Indexing:
        arr = store[tweet_id]   # np.ndarray (img_size, img_size, 3) uint8
                                # or None if tweet_id not in store
        tweet_id in store       # O(1) hash lookup
    """

    def __init__(
        self,
        bin_path: Path,
        index: dict[str, int],
        n_images: int,
        img_size: int,
    ) -> None:
        self._bin_path = bin_path
        self._index = index          # {tweet_id: row_idx}
        self._n = n_images
        self._img_size = img_size
        self._arr: np.memmap = self._open_memmap()

    def _open_memmap(self) -> np.memmap:
        return np.memmap(
            self._bin_path,
            dtype=np.uint8,
            mode="r",
            shape=(self._n, self._img_size, self._img_size, 3),
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def __contains__(self, tweet_id: str) -> bool:
        return tweet_id in self._index

    def __getitem__(self, tweet_id: str) -> Optional[np.ndarray]:
        """Return a (H, W, 3) uint8 copy, or None if not found."""
        idx = self._index.get(tweet_id)
        if idx is None:
            return None
        # np.array() copies the memmap view — avoids PIL issues with read-only views
        return np.array(self._arr[idx], dtype=np.uint8)

    def __len__(self) -> int:
        return self._n

    def get(self, tweet_id: str, default=None) -> Optional[np.ndarray]:
        result = self[tweet_id]
        return result if result is not None else default

    # ── Pickle support (for DataLoader num_workers > 0) ──────────────────────
    # Each worker re-opens the file independently — no 22GB piped over a pipe.

    def __getstate__(self) -> dict:
        return {
            "_bin_path": self._bin_path,
            "_index":    self._index,
            "_n":        self._n,
            "_img_size": self._img_size,
        }

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._arr = self._open_memmap()

    def __repr__(self) -> str:
        size_gb = self._bin_path.stat().st_size / 1e9 if self._bin_path.exists() else 0
        return (
            f"ImageStore(n={self._n:,}, img_size={self._img_size}, "
            f"file_size={size_gb:.1f}GB)"
        )


# ── Public functions ─────────────────────────────────────────────────────────

def load_image_store(img_size: int = 224) -> Optional[ImageStore]:
    """Load the pre-built image store for the given image size.

    Returns None if the store has not been built yet, so callers can
    fall back to PIL.Image.open() gracefully without crashing.

    Results are cached per-process; repeated calls are free.
    """
    if img_size in _store_cache:
        return _store_cache[img_size]

    bp = _bin_path(img_size)
    ip = _idx_path(img_size)

    if not bp.exists() or not ip.exists():
        return None

    with open(ip, "r", encoding="utf-8") as f:
        index: dict[str, int] = json.load(f)

    n = len(index)
    store = ImageStore(bp, index, n, img_size)
    _store_cache[img_size] = store
    logger.info(
        f"[ImageStore] Loaded: {n:,} images  "
        f"({bp.stat().st_size/1e9:.1f}GB on disk)  img_size={img_size}"
    )
    return store


def build_image_store(
    img_dir: Optional[Path] = None,
    img_size: int = 224,
    force: bool = False,
) -> ImageStore:
    """Build the numpy memmap image store from JPEG files.

    One-time operation (~5–10 min for 150k images).  Creates two files:
        dataset/image_store_<img_size>.bin        — raw uint8 pixels (N×H×W×3)
        dataset/image_store_<img_size>_index.json — {tweet_id: row_idx}

    Args:
        img_dir:  Directory containing <tweet_id>.jpg files.
                  Defaults to dataset/img_resized (from utils.config.IMG_DIR).
        img_size: Expected image side length. Images not matching this size
                  are resized automatically.
        force:    Rebuild even if the files already exist.

    Returns:
        Loaded ImageStore instance.
    """
    if img_dir is None:
        img_dir = Path(IMG_DIR)

    bp = _bin_path(img_size)
    ip = _idx_path(img_size)

    if bp.exists() and ip.exists() and not force:
        logger.info(f"[ImageStore] Already built at {bp}. Loading ...")
        existing = load_image_store(img_size)
        if existing is not None:
            return existing

    # ── Discover images ──────────────────────────────────────────────────────
    img_files = sorted(img_dir.glob("*.jpg"))
    if not img_files:
        raise FileNotFoundError(f"[ImageStore] No .jpg files found in {img_dir}")

    n = len(img_files)
    est_gb = n * img_size * img_size * 3 / 1e9
    logger.info(
        f"[ImageStore] Building: {n:,} images × {img_size}px  "
        f"→ ~{est_gb:.1f}GB  at {bp}"
    )

    # ── Allocate memmap ──────────────────────────────────────────────────────
    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.memmap(bp, dtype=np.uint8, mode="w+", shape=(n, img_size, img_size, 3))

    index: dict[str, int] = {}
    failed = 0
    t0 = time.time()

    for i, img_path in enumerate(img_files):
        tweet_id = img_path.stem
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                if im.size != (img_size, img_size):
                    im = im.resize((img_size, img_size), Image.BILINEAR)
                arr[i] = np.asarray(im, dtype=np.uint8)
            index[tweet_id] = i
        except Exception as e:
            failed += 1
            arr[i] = 0      # zeros = black placeholder for missing images
            logger.debug(f"[ImageStore] Failed {img_path.name}: {e}")

        if (i + 1) % 10_000 == 0:
            arr.flush()     # flush periodically to avoid huge dirty-page spikes
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            logger.info(
                f"[ImageStore] {i+1:,}/{n:,}  "
                f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)"
            )

    arr.flush()
    del arr     # close the write handle

    # ── Save index ───────────────────────────────────────────────────────────
    with open(ip, "w", encoding="utf-8") as f:
        json.dump(index, f)

    total = time.time() - t0
    logger.info(
        f"[ImageStore] Done in {total:.0f}s: "
        f"{len(index):,} stored, {failed} failed  →  {bp} ({bp.stat().st_size/1e9:.1f}GB)"
    )

    # Clear cache so next load_image_store() picks up the fresh file
    _store_cache.pop(img_size, None)
    return load_image_store(img_size)
