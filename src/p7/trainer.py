"""P7 Trainer — two-stage training orchestrator for all MHSDF variations.

Handles:
- Single-stage training (P7-A binary, P7-B 6-class)
- Two-stage training (P7-C single-label, P7-D multilabel)
- Backbone weight transfer from Stage 1 → Stage 2
- Per-epoch validation + best-model checkpointing
- Per-category threshold calibration (P7-D only)
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
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from src.p7.config import P7Config
from src.p7.dataset import HATE_CAT_NAMES, P7Dataset, p7_collate_fn, build_token_cache
from src.p7.losses import get_p7_loss
from src.p7.glove_init import build_glove_embedding_matrix
from src.data.image_store import build_image_store
from src.p7.metrics import (
    apply_thresholds,
    calibrate_thresholds,
    compute_multilabel_metrics,
    compute_pipeline_metrics,
)
from src.p7.model import MHSDF
from src.p7.tokenizer import P7Tokenizer
from src.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics

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


# ── Single-epoch helpers ──────────────────────────────────────────────────────

def _get_targets(batch: dict, variation: str, stage: int) -> torch.Tensor:
    """Extract the correct target tensor for a batch."""
    if variation == "B":
        return batch["label_6class"]
    if variation in ("C", "D") and stage == 2:
        if variation == "D":
            # Multi-label: 6-element → 5-element (hate categories only)
            return batch["multi_label_binary"][:, 1:]   # drop NotHate col
        else:
            return batch["label_s2"]
    # Default: binary (P7-A, Stage 1 of C/D)
    return batch["label_binary"].float()


def _train_epoch(
    model: MHSDF,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: P7Config,
    variation: str,
    stage: int,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> dict:
    model.train()
    total_loss = 0.0
    n_steps = 0
    use_amp = scaler is not None and device.type == "cuda"

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        token_ids = batch["token_ids"].to(device)
        targets = _get_targets(batch, variation, stage).to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images, token_ids)

        # Compute loss in FP32 — BCEWithLogitsLoss with large pos_weight
        # overflows in FP16, producing NaN gradients that poison training.
        logits_f32 = logits.float()
        if variation == "A" or (variation in ("C", "D") and stage == 1):
            loss = criterion(logits_f32.squeeze(-1), targets)
        else:
            loss = criterion(logits_f32, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        total_loss += loss.item()
        n_steps += 1

        if (step + 1) % config.log_every_n_steps == 0:
            logger.info(
                f"  [Epoch {epoch+1}] step {step+1}/{len(loader)}  "
                f"loss={total_loss/n_steps:.4f}"
            )

    return {"train/loss": total_loss / max(n_steps, 1)}


@torch.no_grad()
def _eval_epoch(
    model: MHSDF,
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
        images    = batch["image"].to(device)
        token_ids = batch["token_ids"].to(device)
        logits    = model(images, token_ids)
        targets   = _get_targets(batch, variation, stage).to(device)

        # Compute val loss in FP32 (same as train — consistent comparison)
        logits_f32 = logits.float()
        if variation == "A" or (variation in ("C", "D") and stage == 1):
            loss = criterion(logits_f32.squeeze(-1), targets)
        else:
            loss = criterion(logits_f32, targets)
        total_val_loss += loss.item()
        n_val_steps += 1

        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    logits_arr  = torch.cat(all_logits,  dim=0).numpy()
    targets_arr = torch.cat(all_targets, dim=0).numpy()
    metrics = {"val/loss": total_val_loss / max(n_val_steps, 1)}

    # ── Binary (P7-A, Stage 1 of C/D) ────────────────────────────────────
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        probs = torch.sigmoid(torch.tensor(logits_arr)).squeeze(-1).numpy()
        preds = (probs >= 0.5).astype(int)
        metrics.update(compute_binary_metrics(targets_arr.astype(int), preds, y_prob=probs))

    # ── P7-B: direct 6-class ─────────────────────────────────────────────
    elif variation == "B":
        preds = np.argmax(logits_arr, axis=-1)
        probs = torch.softmax(torch.tensor(logits_arr), dim=-1).numpy()
        metrics.update(compute_multiclass_metrics(targets_arr.astype(int), preds, y_prob=probs))

    # ── P7-C Stage 2: 5-class single-label ───────────────────────────────
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

    # ── P7-D Stage 2: 5-way multilabel ───────────────────────────────────
    elif variation == "D" and stage == 2:
        if thresholds is None:
            thresholds = np.full(logits_arr.shape[1], 0.5)
        preds = apply_thresholds(logits_arr, thresholds)
        metrics.update(compute_multilabel_metrics(targets_arr.astype(int), preds, HATE_CAT_NAMES))

    return metrics, logits_arr, targets_arr


# ── Single-stage trainer ──────────────────────────────────────────────────────

def train_stage(
    model: MHSDF,
    train_ds: P7Dataset,
    val_ds: P7Dataset,
    config: P7Config,
    device: torch.device,
    variation: str,
    stage: int,
    epochs: int,
    lr: float,
    batch_size: int,
    save_dir: Path,
) -> tuple[MHSDF, dict, np.ndarray, np.ndarray]:
    """Train a single stage; return best model, metrics, val logits & targets."""

    # num_workers: memmap store is picklable (re-opens file in worker) so workers are safe.
    # Only the legacy RAM dict cache (_image_cache) is incompatible with workers.
    _use_dict_cache = getattr(config, "use_image_cache", False)
    _num_workers = 0 if _use_dict_cache else config.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_num_workers,
        collate_fn=p7_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(_num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=_num_workers,
        collate_fn=p7_collate_fn,
        persistent_workers=(_num_workers > 0),
    )

    criterion = get_p7_loss(
        variation=variation,
        stage=stage,
        train_dataset=train_ds,
        device=device,
        focal_gamma=config.s1_focal_gamma,
        s1_loss_type=config.s1_loss,
        label_smoothing=getattr(config, "label_smoothing", 0.0),
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)

    # AMP scaler — enabled only when use_amp=True and device is CUDA
    scaler = (
        torch.cuda.amp.GradScaler()
        if getattr(config, "use_amp", False) and device.type == "cuda"
        else None
    )
    if scaler is not None:
        logger.info("[P7 Trainer] AMP (FP16) enabled.")

    # ── torch.compile (PyTorch 2.x JIT optimisation) ──────────────────────
    # NOTE: requires Linux + Triton. Disabled by default on Windows.
    if getattr(config, "use_compile", False) and device.type == "cuda":
        try:
            compiled = torch.compile(model)
            # Force eager compilation NOW with a dummy batch so lazy errors
            # surface here (not mid-epoch with a cryptic BackendCompilerFailed).
            _dummy_img = torch.zeros(1, 3, config.img_size, config.img_size, device=device)
            _dummy_tok = torch.zeros(1, config.max_seq_len, dtype=torch.long, device=device)
            with torch.no_grad():
                compiled(_dummy_img, _dummy_tok)
            model = compiled
            logger.info("[P7 Trainer] torch.compile enabled.")
        except Exception as e:
            logger.warning(f"[P7 Trainer] torch.compile failed ({type(e).__name__}). Using eager mode.")

    if config.scheduler == "cosine":
        warmup_steps = int(epochs * len(train_loader) * config.warmup_ratio)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    elif config.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    else:
        scheduler = None

    best_metric = -1.0
    best_state  = None
    history     = []
    no_improve  = 0
    patience    = getattr(config, "early_stop_patience", 0)

    # ── DataParallel: use all visible CUDA GPUs ────────────────────────────
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if getattr(config, "use_data_parallel", False) and n_gpus > 1:
        model = nn.DataParallel(model)
        logger.info(f"[P7 Trainer] DataParallel: using {n_gpus} GPUs.")

    # Determine primary metric key for best-model selection
    if variation == "A" or (variation in ("C", "D") and stage == 1):
        primary_key = "binary/macro_f1"
    elif variation == "B":
        primary_key = "multiclass/macro_f1"
    elif variation == "C" and stage == 2:
        primary_key = "multiclass/macro_f1"
    else:  # D Stage 2
        primary_key = "multilabel/macro_f1"

    logger.info(
        f"\n[P7 Trainer] variation={variation}  stage={stage}  "
        f"epochs={epochs}  lr={lr}  batch={batch_size}  device={device}"
    )

    for epoch in range(epochs):
        t0 = time.time()

        train_metrics = _train_epoch(
            model, train_loader, criterion, optimizer, device,
            config, variation, stage, epoch, scaler,
        )

        val_metrics, val_logits, val_targets = _eval_epoch(
            model, val_loader, device, variation, stage,
            criterion=criterion,
        )

        if scheduler is not None:
            if config.scheduler == "cosine":
                scheduler.step()
            elif config.scheduler == "step":
                scheduler.step()

        epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch + 1}
        history.append(epoch_metrics)

        primary_val = val_metrics.get(primary_key, 0.0)
        # ── State dict save: unwrap DataParallel if present ──────────────────
        _module = model.module if isinstance(model, nn.DataParallel) else model
        if primary_val > best_metric:
            best_metric = primary_val
            best_state  = {k: v.clone() for k, v in _module.state_dict().items()}
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_dir / f"stage{stage}_best.pt")
            logger.info(f"  ↑ New best {primary_key}={best_metric:.4f}  (saved)")
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                logger.info(
                    f"[P7 Trainer] Early stopping: no improvement for "
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

    # Restore best weights into the underlying module
    _module = model.module if isinstance(model, nn.DataParallel) else model
    if best_state is not None:
        _module.load_state_dict(best_state)
    # Always return the unwrapped MHSDF (not a DataParallel wrapper)
    return _module, {"best": best_metric, "history": history}, val_logits, val_targets


# ── Main P7 training entry point ─────────────────────────────────────────────

def run_p7(config: P7Config) -> dict:
    """Run the full P7 pipeline for the given config.

    Handles all 4 variations (A/B/C/D) and all 3 text modes.

    Args:
        config: P7Config instance describing the experiment.

    Returns:
        Dict with all final metrics for this run.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info(f"[P7] Starting: {config.run_name}  device={device}")

    # ── GPU tuning ────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # finds fastest conv algorithm

    # ── Paths ────────────────────────────────────────────────────────────
    ckpt_dir = Path(config.checkpoint_dir) / config.run_name
    results_dir = Path(config.results_dir) / config.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = P7Tokenizer(
        model_name=config.bert_model_name,
        max_seq_len=config.max_seq_len,
    )
    vocab_size = tokenizer.vocab_size

    # ── Image store (build if not already built) ─────────────────────────
    if getattr(config, "use_image_store", False):
        from src.utils.config import IMG_DIR
        build_image_store(img_size=config.img_size)

    # ── GloVe embeddings (optional) ───────────────────────────────────────
    pretrained_embeddings = None
    if getattr(config, "use_glove", False) and config.glove_path:
        pretrained_embeddings = build_glove_embedding_matrix(
            glove_path=config.glove_path,
            tokenizer=tokenizer,
            embed_dim=config.glove_dim,
        )
        # embed_dim must match glove_dim when using GloVe
        if config.embed_dim != config.glove_dim:
            logger.warning(
                f"[P7] embed_dim={config.embed_dim} != glove_dim={config.glove_dim}. "
                "Set embed_dim=glove_dim in P7Config for correct GloVe init."
            )

    # ── Token cache (pre-tokenize once, skip HF tokenizer per sample) ────
    tok_cache = build_token_cache(
        text_mode=config.text_mode,
        max_seq_len=config.max_seq_len,
        tokenizer=tokenizer,
    )

    all_metrics = {"config": config.run_name, "variation": config.variation}

    # ══════════════════════════════════════════════════════════════════════
    # P7-A: Binary classification (Stage 1 standalone)
    # ══════════════════════════════════════════════════════════════════════
    if config.variation == "A":
        train_ds = P7Dataset("train", config, tokenizer, stage=None, is_training=True,
                             max_samples=config.max_train_samples, token_cache=tok_cache)
        val_ds   = P7Dataset("val",   config, tokenizer, stage=None, is_training=False,
                             max_samples=config.max_val_samples,   token_cache=tok_cache)
        model = MHSDF.from_config(config, vocab_size, num_classes=config.num_classes_s1,
                                   pretrained_embeddings=pretrained_embeddings).to(device)

        model, stage_metrics, _, _ = train_stage(
            model, train_ds, val_ds, config, device,
            variation="A", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
        )
        all_metrics["stage1"] = stage_metrics

    # ══════════════════════════════════════════════════════════════════════
    # P7-B: Direct 6-class (single stage)
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "B":
        train_ds = P7Dataset("train", config, tokenizer, stage=None, is_training=True,
                             max_samples=config.max_train_samples, token_cache=tok_cache)
        val_ds   = P7Dataset("val",   config, tokenizer, stage=None, is_training=False,
                             max_samples=config.max_val_samples,   token_cache=tok_cache)
        model = MHSDF.from_config(config, vocab_size, num_classes=6,
                                   pretrained_embeddings=pretrained_embeddings).to(device)

        model, stage_metrics, _, _ = train_stage(
            model, train_ds, val_ds, config, device,
            variation="B", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
        )
        all_metrics["stage1"] = stage_metrics

    # ══════════════════════════════════════════════════════════════════════
    # P7-C: Two-stage, single-label
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "C":
        # — Stage 1: binary on full dataset
        train_s1 = P7Dataset("train", config, tokenizer, stage=1, is_training=True,
                             max_samples=config.max_train_samples, token_cache=tok_cache)
        val_s1   = P7Dataset("val",   config, tokenizer, stage=1, is_training=False,
                             max_samples=config.max_val_samples,   token_cache=tok_cache)
        model_s1 = MHSDF.from_config(config, vocab_size, num_classes=config.num_classes_s1,
                                      pretrained_embeddings=pretrained_embeddings).to(device)

        model_s1, s1_metrics, s1_val_logits, s1_val_targets = train_stage(
            model_s1, train_s1, val_s1, config, device,
            variation="C", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
        )
        all_metrics["stage1"] = s1_metrics

        # — Stage 2: 5-class on hateful-only
        train_s2 = P7Dataset("train", config, tokenizer, stage=2, is_training=True,
                             token_cache=tok_cache)
        val_s2   = P7Dataset("val",   config, tokenizer, stage=2, is_training=False,
                             token_cache=tok_cache)
        model_s2 = MHSDF.from_config(config, vocab_size, num_classes=5,
                                      pretrained_embeddings=pretrained_embeddings).to(device)
        model_s2.transfer_backbone(model_s1)   # copy CNN + BiLSTM weights

        model_s2, s2_metrics, s2_val_logits, s2_val_targets = train_stage(
            model_s2, train_s2, val_s2, config, device,
            variation="C", stage=2,
            epochs=config.s2_epochs, lr=config.s2_lr,
            batch_size=config.s2_batch_size, save_dir=ckpt_dir,
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
    # P7-D: Two-stage, multi-label (primary)
    # ══════════════════════════════════════════════════════════════════════
    elif config.variation == "D":
        # — Stage 1: binary on full dataset
        train_s1 = P7Dataset("train", config, tokenizer, stage=1, is_training=True,
                             max_samples=config.max_train_samples, token_cache=tok_cache)
        val_s1   = P7Dataset("val",   config, tokenizer, stage=1, is_training=False,
                             max_samples=config.max_val_samples,   token_cache=tok_cache)
        model_s1 = MHSDF.from_config(config, vocab_size, num_classes=config.num_classes_s1,
                                      pretrained_embeddings=pretrained_embeddings).to(device)

        model_s1, s1_metrics, s1_val_logits, s1_val_targets = train_stage(
            model_s1, train_s1, val_s1, config, device,
            variation="D", stage=1,
            epochs=config.s1_epochs, lr=config.s1_lr,
            batch_size=config.s1_batch_size, save_dir=ckpt_dir,
        )
        all_metrics["stage1"] = s1_metrics

        # — Stage 2: 5-way multilabel on hateful-only
        train_s2 = P7Dataset("train", config, tokenizer, stage=2, is_training=True,
                             token_cache=tok_cache)
        val_s2   = P7Dataset("val",   config, tokenizer, stage=2, is_training=False,
                             token_cache=tok_cache)
        model_s2 = MHSDF.from_config(config, vocab_size, num_classes=5,
                                      pretrained_embeddings=pretrained_embeddings).to(device)
        model_s2.transfer_backbone(model_s1)

        model_s2, s2_metrics, s2_val_logits, s2_val_targets = train_stage(
            model_s2, train_s2, val_s2, config, device,
            variation="D", stage=2,
            epochs=config.s2_epochs, lr=config.s2_lr,
            batch_size=config.s2_batch_size, save_dir=ckpt_dir,
        )
        all_metrics["stage2"] = s2_metrics

        # — Threshold calibration
        logger.info("[P7-D] Calibrating per-category thresholds on val set...")
        thresholds = calibrate_thresholds(s2_val_logits, s2_val_targets.astype(int))
        np.save(ckpt_dir / "stage2_thresholds.npy", thresholds)
        all_metrics["thresholds"] = thresholds.tolist()
        logger.info(f"[P7-D] Thresholds: {dict(zip(HATE_CAT_NAMES, thresholds.tolist()))}")

        # Final multilabel metrics with calibrated thresholds
        preds_cal = apply_thresholds(s2_val_logits, thresholds)
        cal_metrics = compute_multilabel_metrics(
            s2_val_targets.astype(int), preds_cal, HATE_CAT_NAMES
        )
        all_metrics["stage2_calibrated"] = cal_metrics
        logger.info(
            f"[P7-D] Calibrated multilabel macro_f1={cal_metrics.get('multilabel/macro_f1', 0):.4f}"
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
    logger.info(f"[P7] Done. Results → {results_dir}")

    return all_metrics
