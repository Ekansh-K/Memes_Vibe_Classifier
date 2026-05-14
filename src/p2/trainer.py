"""P2 Trainer — two-stage training orchestrator for all TCAM variations.

Handles:
- Single-stage training (P2-A binary, P2-B 6-class)
- Two-stage training (P2-C single-label, P2-D multilabel)
- Stage 2 fusion reinit (keeps proj_t, resets cross_attn + head)
- Per-epoch validation + best-model checkpointing
- Per-category threshold calibration (P2-D only)
- DataParallel across 2× Kaggle T4 GPUs
- JSON metrics logging
"""

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score as sklearn_f1_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.p2.config import P2Config
from src.p2.dataset import HATE_CAT_NAMES, P2Dataset, p2_collate_fn
from src.p2.losses import get_p2_loss
from src.p2.metrics import (
    apply_thresholds,
    calibrate_thresholds,
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_multilabel_metrics,
    compute_pipeline_metrics,
)
from src.p2.model import TCAM

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def _get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine schedule with linear warmup (step-level)."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ── Helpers ──────────────────────────────────────────────────────────────

def _get_model_module(model) -> "TCAM":
    """Unwrap nn.DataParallel to get the underlying TCAM module."""
    return model.module if isinstance(model, nn.DataParallel) else model


def _preprocess_images_for_trainer(
    pil_images: list, model, device: torch.device
) -> torch.Tensor:
    """Preprocess PIL images to tensor using CLIP's preprocess transform.

    Must be called OUTSIDE model.forward() so that DataParallel receives
    a proper Tensor to scatter across GPUs — lists cannot be scattered.

    Args:
        pil_images: list of B PIL.Image objects.
        model:      TCAM or nn.DataParallel(TCAM).
        device:     Target device.

    Returns:
        (B, 3, 224, 224) float32 tensor on device.
    """
    module = _get_model_module(model)
    tensors = [module.clip_preprocess(img) for img in pil_images]
    return torch.stack(tensors).to(device)


def _get_targets(batch: dict, variation: str, stage: int) -> torch.Tensor:
    """Extract the correct target tensor for a batch.

    Label map:
        P2-A, Stage 1 of C/D  → label_binary (float, 0.0 or 1.0)
        P2-B                  → label_6class  (long, 0–5)
        P2-C Stage 2          → label_s2      (long, 0–4 hate categories)
        P2-D Stage 2          → multi_label_binary[:, 1:] (float, 5 hate cols)
    """
    if variation == "B":
        return batch["label_6class"]
    if variation in ("C", "D") and stage == 2:
        if variation == "D":
            # Drop col 0 (NotHate) → 5 hate-category columns
            return batch["multi_label_binary"][:, 1:]
        else:
            return batch["label_s2"]
    # Default: binary (P2-A, Stage 1 of C/D)
    return batch["label_binary"].float()


# ── Single-epoch helpers ──────────────────────────────────────────────────────

def _train_epoch(
    model,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: P2Config,
    variation: str,
    stage: int,
    epoch: int,
    scaler: Optional[torch.amp.GradScaler] = None,
    scheduler=None,
) -> dict:
    model.train()
    # Keep frozen encoders in eval mode even during training
    _module = _get_model_module(model)
    _module._clip_model.eval()
    _module.tweet_encoder.eval()

    total_loss = 0.0
    n_steps = 0
    use_amp = scaler is not None and device.type == "cuda"
    accum_steps = max(1, config.grad_accum_steps)

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        pil_images = batch["image"]       # list of PIL images
        texts      = batch["text"]         # list of strings
        targets    = _get_targets(batch, variation, stage).to(device)

        # Preprocess images to tensor BEFORE model forward so DataParallel
        # can scatter the tensor across GPUs (lists are not scatter-able).
        images_tensor = _preprocess_images_for_trainer(pil_images, model, device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images_tensor, texts)

        # Compute loss in FP32 for numerical stability
        logits_f32 = logits.float()
        if variation == "A" or (variation in ("C", "D") and stage == 1):
            loss = criterion(logits_f32.squeeze(-1), targets)
        else:
            loss = criterion(logits_f32, targets)

        loss = loss / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps
        n_steps += 1

        is_last_batch = (step + 1 == len(loader))
        should_step = ((step + 1) % accum_steps == 0) or is_last_batch

        if should_step:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        if (step + 1) % config.log_every_n_steps == 0:
            logger.info(
                f"  [Epoch {epoch+1}] step {step+1}/{len(loader)}  "
                f"loss={total_loss/n_steps:.4f}"
            )

    return {"train/loss": total_loss / max(n_steps, 1)}


@torch.no_grad()
def _eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    variation: str,
    stage: int,
    criterion: nn.Module,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate one epoch; return (metrics_dict, all_logits, all_targets)."""
    model.eval()
    all_logits, all_targets = [], []
    total_val_loss = 0.0
    n_val_steps = 0

    for batch in loader:
        pil_images = batch["image"]
        texts      = batch["text"]
        targets    = _get_targets(batch, variation, stage).to(device)

        # Same DataParallel fix as train: preprocess to tensor before forward
        images_tensor = _preprocess_images_for_trainer(pil_images, model, device)
        logits = model(images_tensor, texts)

        logits_f32 = logits.float()
        if variation == "A" or (variation in ("C", "D") and stage == 1):
            loss = criterion(logits_f32.squeeze(-1), targets)
        else:
            loss = criterion(logits_f32, targets)
        total_val_loss += loss.item()
        n_val_steps += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits_arr = torch.cat(all_logits, dim=0).numpy()
    targets_arr = torch.cat(all_targets, dim=0).numpy()
    metrics = {"val/loss": total_val_loss / max(n_val_steps, 1)}

    # ── Binary (P2-A, Stage 1 of C/D) ────────────────────────────────────
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        probs = torch.sigmoid(torch.tensor(logits_arr)).squeeze(-1).numpy()
        preds = (probs >= 0.5).astype(int)
        metrics.update(compute_binary_metrics(targets_arr.astype(int), preds, y_prob=probs))

    # ── P2-B: direct 6-class ─────────────────────────────────────────────
    elif variation == "B":
        preds = np.argmax(logits_arr, axis=-1)
        probs = torch.softmax(torch.tensor(logits_arr), dim=-1).numpy()
        metrics.update(compute_multiclass_metrics(targets_arr.astype(int), preds, y_prob=probs))

    # ── P2-C Stage 2: 5-class single-label ───────────────────────────────
    elif variation == "C" and stage == 2:
        preds = np.argmax(logits_arr, axis=-1)
        probs = torch.softmax(torch.tensor(logits_arr), dim=-1).numpy()
        metrics.update(
            compute_multiclass_metrics(
                targets_arr.astype(int), preds,
                y_prob=probs,
                class_names=HATE_CAT_NAMES,
            )
        )

    # ── P2-D Stage 2: 5-way multilabel ───────────────────────────────────
    elif variation == "D" and stage == 2:
        if thresholds is None:
            thresholds = np.full(logits_arr.shape[1], 0.5)
        preds = apply_thresholds(logits_arr, thresholds)
        metrics.update(compute_multilabel_metrics(targets_arr.astype(int), preds, HATE_CAT_NAMES))

    return metrics, logits_arr, targets_arr


# ── Single-stage trainer ──────────────────────────────────────────────────────

def train_stage(
    model,
    train_ds: P2Dataset,
    val_ds: P2Dataset,
    config: P2Config,
    device: torch.device,
    variation: str,
    stage: int,
    epochs: int,
    lr: float,
    batch_size: int,
    save_dir: Path,
    warmup_ratio: float = 0.05,
) -> tuple:
    """Train a single stage; return best model, metrics, val logits & targets."""

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=p2_collate_fn,
        pin_memory=False,  # PIL images can't be pinned
        persistent_workers=(config.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=p2_collate_fn,
        persistent_workers=(config.num_workers > 0),
    )

    criterion = get_p2_loss(
        variation=variation,
        stage=stage,
        train_dataset=train_ds,
        device=device,
        label_smoothing=config.label_smoothing,
    )

    # Trainable params only — CLIP and TweetEval excluded automatically
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"[P2 Trainer] Trainable params: {n_trainable:,}")
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=config.weight_decay)

    # AMP scaler
    scaler = (
        torch.amp.GradScaler("cuda")
        if config.use_amp and device.type == "cuda"
        else None
    )
    if scaler is not None:
        logger.info("[P2 Trainer] AMP (FP16) enabled.")

    # Step-level cosine schedule with warmup
    total_steps = (len(train_loader) * epochs) // max(1, config.grad_accum_steps)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(
        f"[P2 Trainer] Scheduler: {warmup_steps} warmup steps / "
        f"{total_steps} total steps (warmup_ratio={warmup_ratio})"
    )

    # DataParallel — wrap AFTER optimizer is set (optimizer holds refs to params)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if config.use_data_parallel and n_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"[P2 Trainer] DataParallel: using {n_gpus} GPUs. "
                    f"Images preprocessed to tensors before scatter.")

    # Primary metric for best-model selection
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        primary_key = "binary/macro_f1"
    elif variation == "B":
        primary_key = "multiclass/macro_f1"
    elif variation == "C" and stage == 2:
        primary_key = "multiclass/macro_f1"
    else:  # D Stage 2
        primary_key = "multilabel/macro_f1"

    best_metric = -1.0
    best_state = None
    history = []
    no_improve = 0
    patience = config.early_stop_patience

    logger.info(
        f"\n[P2 Trainer] variation={variation}  stage={stage}  "
        f"epochs={epochs}  lr={lr}  batch={batch_size}  "
        f"grad_accum={config.grad_accum_steps}  device={device}"
    )

    for epoch in range(epochs):
        t0 = time.time()

        train_metrics = _train_epoch(
            model, train_loader, criterion, optimizer, device,
            config, variation, stage, epoch, scaler, scheduler,
        )

        val_metrics, val_logits, val_targets = _eval_epoch(
            model, val_loader, device, variation, stage,
            criterion=criterion,
        )

        # Stage 1 binary threshold calibration
        if primary_key == "binary/macro_f1" and len(val_logits) > 0:
            probs = 1.0 / (1.0 + np.exp(-val_logits.squeeze(-1)))
            raw_f1 = val_metrics.get("binary/macro_f1", 0.0)
            best_t, best_f1 = 0.5, raw_f1
            for _t in np.arange(0.10, 0.91, 0.01):
                _preds = (probs >= _t).astype(int)
                _f1 = float(sklearn_f1_score(
                    val_targets.astype(int), _preds,
                    average="macro", zero_division=0
                ))
                if _f1 > best_f1:
                    best_f1, best_t = _f1, float(_t)
            val_metrics["binary/macro_f1"] = best_f1
            val_metrics["binary/macro_f1_raw"] = raw_f1
            val_metrics["binary/threshold"] = best_t
            logger.info(
                f"  [Stage1 Cal] threshold={best_t:.2f}  "
                f"calibrated_f1={best_f1:.4f}  raw_f1@0.5={raw_f1:.4f}"
            )

        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        history.append(epoch_metrics)

        primary_val = val_metrics.get(primary_key, 0.0)
        _module = model.module if isinstance(model, nn.DataParallel) else model

        if primary_val > best_metric:
            best_metric = primary_val
            # Only save trainable params (proj_t, cross_attn, head)
            best_state = {
                k: v.clone() for k, v in _module.state_dict().items()
                if any(
                    k.startswith(prefix)
                    for prefix in ["proj_t.", "cross_attn.", "head."]
                )
            }
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_dir / f"stage{stage}_best.pt")
            logger.info(f"  ↑ New best {primary_key}={best_metric:.4f}  (saved)")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                logger.info(
                    f"[P2 Trainer] Early stopping: no improvement for "
                    f"{patience} epochs at epoch {epoch+1}."
                )
                break

        elapsed = time.time() - t0
        val_loss = val_metrics.get("val/loss", float("nan"))
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_metrics['train/loss']:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"{primary_key}={primary_val:.4f}  "
            f"({elapsed:.1f}s)"
        )

    # Restore best weights
    _module = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        _module.load_state_dict(best_state, strict=False)

    return _module, {"best": best_metric, "history": history}, val_logits, val_targets


# ── Main P2 training entry point ─────────────────────────────────────────────

def run_p2(config: P2Config) -> dict:
    """Run the full P2 pipeline for the given config.

    Handles all 4 variations (A/B/C/D) and both text modes.

    Args:
        config: P2Config instance describing the experiment.

    Returns:
        Dict with all final metrics for this run.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info(f"[P2] Starting: {config.run_name}  device={device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        n_gpus = torch.cuda.device_count()
        logger.info(f"[P2] CUDA GPUs available: {n_gpus}")

    # ── Paths ────────────────────────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir) / config.run_name
    results_dir = Path(config.results_dir) / config.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Image store (build if not exists) ────────────────────────────────
    if config.use_image_store:
        try:
            from src.data.image_store import build_image_store
            build_image_store(img_size=config.img_size)
        except Exception as e:
            logger.warning(f"[P2] Image store build failed: {e}. Using disk loading.")

    all_metrics = {"config": config.run_name, "variation": config.variation}

    # ══════════════════════════════════════════════════════════════════════
    # P2-A: Binary classification (Stage 1 standalone)
    # ══════════════════════════════════════════════════════════════════════
    if config.variation == "A":
        train_ds = P2Dataset("train", config, stage=None, is_training=True,
                             max_samples=config.max_train_samples)
        val_ds = P2Dataset("val", config, stage=None, is_training=False,
                           max_samples=config.max_val_samples)
        model = TCAM.from_config(config, num_classes=config.num_classes_s1).to(device)

        model, stage_metrics, _, _ = train_stage(
            model, train_ds, val_ds, config, device,
            variation="A", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s1_warmup_ratio,
        )
        all_metrics["stage1"] = stage_metrics

    # ══════════════════════════════════════════════════════════════════════
    # P2-B: Direct 6-class (single stage)
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "B":
        train_ds = P2Dataset("train", config, stage=None, is_training=True,
                             max_samples=config.max_train_samples)
        val_ds = P2Dataset("val", config, stage=None, is_training=False,
                           max_samples=config.max_val_samples)
        model = TCAM.from_config(config, num_classes=6).to(device)

        model, stage_metrics, _, _ = train_stage(
            model, train_ds, val_ds, config, device,
            variation="B", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s1_warmup_ratio,
        )
        all_metrics["stage1"] = stage_metrics

    # ══════════════════════════════════════════════════════════════════════
    # P2-C: Two-stage, single-label
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "C":
        # Stage 1: binary on full dataset
        train_s1 = P2Dataset("train", config, stage=1, is_training=True,
                             max_samples=config.max_train_samples)
        val_s1 = P2Dataset("val", config, stage=1, is_training=False,
                           max_samples=config.max_val_samples)
        model = TCAM.from_config(config, num_classes=config.num_classes_s1).to(device)

        model, s1_metrics, s1_val_logits, s1_val_targets = train_stage(
            model, train_s1, val_s1, config, device,
            variation="C", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s1_warmup_ratio,
        )
        all_metrics["stage1"] = s1_metrics

        # Stage 2: 5-class on hateful-only — reinit cross_attn + head, keep proj_t
        model.reinit_for_stage2(new_num_classes=5)
        model = model.to(device)

        train_s2 = P2Dataset("train", config, stage=2, is_training=True)
        val_s2 = P2Dataset("val", config, stage=2, is_training=False)

        model, s2_metrics, s2_val_logits, s2_val_targets = train_stage(
            model, train_s2, val_s2, config, device,
            variation="C", stage=2,
            epochs=config.s2_epochs, lr=config.s2_lr,
            batch_size=config.s2_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s2_warmup_ratio,
        )
        all_metrics["stage2"] = s2_metrics

        # Pipeline composite metric
        s1_probs = torch.sigmoid(torch.tensor(s1_val_logits)).squeeze(-1).numpy()
        s1_preds = (s1_probs >= 0.5).astype(int)
        s2_preds = np.argmax(s2_val_logits, axis=-1)
        all_metrics["pipeline"] = compute_pipeline_metrics(
            s1_val_targets.astype(int), s1_preds,
            s2_val_targets.astype(int), s2_preds,
            multilabel=False,
        )

    # ══════════════════════════════════════════════════════════════════════
    # P2-D: Two-stage, multi-label (primary)
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "D":
        # Stage 1: binary on full dataset
        train_s1 = P2Dataset("train", config, stage=1, is_training=True,
                             max_samples=config.max_train_samples)
        val_s1 = P2Dataset("val", config, stage=1, is_training=False,
                           max_samples=config.max_val_samples)
        model = TCAM.from_config(config, num_classes=config.num_classes_s1).to(device)

        model, s1_metrics, s1_val_logits, s1_val_targets = train_stage(
            model, train_s1, val_s1, config, device,
            variation="D", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s1_warmup_ratio,
        )
        all_metrics["stage1"] = s1_metrics

        # Stage 2: 5-way multilabel on hateful-only
        model.reinit_for_stage2(new_num_classes=5)
        model = model.to(device)

        train_s2 = P2Dataset("train", config, stage=2, is_training=True)
        val_s2 = P2Dataset("val", config, stage=2, is_training=False)

        model, s2_metrics, s2_val_logits, s2_val_targets = train_stage(
            model, train_s2, val_s2, config, device,
            variation="D", stage=2,
            epochs=config.s2_epochs, lr=config.s2_lr,
            batch_size=config.s2_batch_size, save_dir=ckpt_dir,
            warmup_ratio=config.s2_warmup_ratio,
        )
        all_metrics["stage2"] = s2_metrics

        # Threshold calibration
        logger.info("[P2-D] Calibrating per-category thresholds on val set...")
        thresholds = calibrate_thresholds(s2_val_logits, s2_val_targets.astype(int))
        np.save(ckpt_dir / "stage2_thresholds.npy", thresholds)
        all_metrics["thresholds"] = thresholds.tolist()
        logger.info(f"[P2-D] Thresholds: {dict(zip(HATE_CAT_NAMES, thresholds.tolist()))}")

        # Final multilabel metrics with calibrated thresholds
        preds_cal = apply_thresholds(s2_val_logits, thresholds)
        cal_metrics = compute_multilabel_metrics(
            s2_val_targets.astype(int), preds_cal, HATE_CAT_NAMES
        )
        all_metrics["stage2_calibrated"] = cal_metrics
        logger.info(
            f"[P2-D] Calibrated multilabel macro_f1="
            f"{cal_metrics.get('multilabel/macro_f1', 0):.4f}"
        )

        # Pipeline composite metric
        s1_probs = torch.sigmoid(torch.tensor(s1_val_logits)).squeeze(-1).numpy()
        s1_preds = (s1_probs >= 0.5).astype(int)
        all_metrics["pipeline"] = compute_pipeline_metrics(
            s1_val_targets.astype(int), s1_preds,
            s2_val_targets.astype(int), preds_cal,
            multilabel=True,
        )

    else:
        raise ValueError(f"Unknown variation: {config.variation!r}")

    # ── Save results ─────────────────────────────────────────────────────
    save_metrics(all_metrics, results_dir / "metrics.json")
    config.save(results_dir / "config.yaml")
    logger.info(f"[P2] Done. Results → {results_dir}")

    return all_metrics
