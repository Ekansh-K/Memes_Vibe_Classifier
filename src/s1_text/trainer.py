"""Train Stage-1 text classifier on soft binary labels."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.p2.losses import SoftBCEWithAgreementWeighting, TemperatureScaler, compute_binary_pos_weight
from src.s1_text.config import S1TextConfig
from src.s1_text.model import TextHateClassifier
from src.stage1.dataset import Stage1Dataset, stage1_collate_text
from src.stage1.eval import evaluate_stage1, save_stage1_results

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _collect_probs(model, loader, device, use_amp, amp_dtype) -> tuple:
    model.eval()
    all_prob, all_true, all_ids = [], [], []
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    for batch in loader:
        with autocast(enabled=use_amp and device.type == "cuda", dtype=dtype):
            logits = model(batch["text"])
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        all_prob.append(probs)
        all_true.append(batch["label_binary"].numpy())
        all_ids.extend(batch["tweet_id"])
    return (
        np.concatenate(all_true),
        np.concatenate(all_prob),
        all_ids,
    )


def run_s1_text(config: S1TextConfig) -> dict:
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    logger.info(f"[S1Text] device={device}  model={config.model_name}  run={config.run_name}")

    train_ds = Stage1Dataset(
        "train",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=False,
        max_samples=config.max_train_samples,
        exclude_full_disagreement=config.exclude_full_disagreement,
        seed=config.seed,
    )
    val_ds = Stage1Dataset(
        "val",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=False,
        max_samples=config.max_val_samples,
        seed=config.seed,
    )

    if len(train_ds) < config.batch_size:
        raise RuntimeError(
            f"Train set size {len(train_ds)} < batch_size {config.batch_size}. "
            "Lower --batch_size or raise --max_train_samples."
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=stage1_collate_text,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= config.batch_size * 2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=stage1_collate_text,
        pin_memory=device.type == "cuda",
    )

    model = TextHateClassifier(
        config.model_name, max_seq_len=config.max_seq_len
    ).to(device)

    hard_labels = [train_ds.labels[i]["hard_label_binary"] for i in train_ds.sample_ids]
    pos_weight = compute_binary_pos_weight(hard_labels, device)
    criterion = SoftBCEWithAgreementWeighting(
        pos_weight=pos_weight,
        agreement_weights=config.agreement_weights,
        use_agreement_weighting=config.use_agreement_weighting,
        label_smoothing=config.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader) // config.grad_accum_steps)
    total_steps = steps_per_epoch * config.epochs
    warmup = int(total_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)

    use_amp = config.use_amp and device.type == "cuda"
    # GradScaler only for fp16; bf16 does not need it
    use_scaler = use_amp and config.amp_dtype == "fp16"
    scaler = GradScaler(enabled=use_scaler)
    amp_dtype = torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16

    best_f1 = -1.0
    best_path = config.run_dir / "best.pt"
    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.results_run_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.run_dir / "config.yaml")
    patience = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"S1Text ep{epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(pbar, 1):
            soft = batch["soft_hate"].to(device)
            hard = batch["label_binary"].to(device).float()
            targets = soft if config.use_soft_labels else hard
            agr = batch["agreement_level"].to(device)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(batch["text"])
            # Loss in fp32 (stable SoftBCE + pos_weight; matches P2 practice)
            loss = criterion(logits.float(), targets, agr) / config.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * config.grad_accum_steps
            pbar.set_postfix(loss=f"{running / step:.4f}")

        y_true, y_prob, ids = _collect_probs(
            model, val_loader, device, use_amp, config.amp_dtype
        )
        metrics = evaluate_stage1(y_true, y_prob)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(step, 1)
        history.append(metrics)
        logger.info(
            f"[S1Text] epoch {epoch}: macro_f1={metrics['macro_f1']:.4f} "
            f"hate_recall={metrics['hate_recall']:.4f} auc={metrics['auc_roc']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "metrics": metrics,
                    "model_name": config.model_name,
                },
                best_path,
            )
            save_stage1_results(
                config.results_run_dir, metrics, y_true, y_prob, ids
            )
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                logger.info(f"[S1Text] Early stop at epoch {epoch}")
                break

    # Reload best + temperature scale
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    y_true, y_prob, ids = _collect_probs(
        model, val_loader, device, use_amp, config.amp_dtype
    )
    # Temperature on logits: refit from probs via inverse sigmoid is awkward;
    # recompute logits for calibration
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(batch["text"])
            all_logits.append(logits.float().cpu())
    logits_cat = torch.cat(all_logits)
    labels_t = torch.tensor(y_true, dtype=torch.float32)
    scaler_t = TemperatureScaler()
    scaler_t.fit(logits_cat, labels_t)
    y_prob_cal = torch.sigmoid(logits_cat / scaler_t.temperature.detach()).numpy()
    final = evaluate_stage1(y_true, y_prob_cal)
    final["history"] = history
    final["temperature"] = float(scaler_t.temperature.item())
    final["run_name"] = config.run_name
    final["model_name"] = config.model_name
    save_stage1_results(config.results_run_dir, final, y_true, y_prob_cal, ids)
    # Persist temperature with checkpoint
    ckpt["temperature"] = final["temperature"]
    ckpt["metrics"] = final
    torch.save(ckpt, best_path)
    logger.info(f"[S1Text] DONE {config.run_name}: macro_f1={final['macro_f1']:.4f}")
    return final
