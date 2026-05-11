"""Image transform pipelines for MMHS150K.

Uses CLIP normalization constants since the primary image encoders
(CLIP ViT, ViT) expect this specific normalization.
"""

from torchvision import transforms

# CLIP normalization — used across all tracks for consistency
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_base_transforms(img_size: int = 224) -> transforms.Compose:
    """Validation/test transforms: resize + normalize (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_train_transforms(
    img_size: int = 224,
    use_random_erasing: bool = True,
) -> transforms.Compose:
    """Training transforms with augmentation.

    Uses scale=(0.8, 1.0) for RandomResizedCrop — aggressive cropping
    (default 0.08-1.0) risks removing text overlays on meme images,
    destroying the OCR signal that's critical for hate speech detection.

    RandomErasing scale=(0.02, 0.15) similarly kept small to preserve text.
    """
    t = [
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
    if use_random_erasing:
        # value=0 erases to black (close to zero-mean after normalise)
        t.append(transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0))
    return transforms.Compose(t)

