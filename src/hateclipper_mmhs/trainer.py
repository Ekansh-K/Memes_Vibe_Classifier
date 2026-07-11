"""Train Hate-CLIPper MMHS Stage-1."""

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

from src.hateclipper_mmhs.config import HateCLIPperConfig
from src.hateclipper_mmhs.model import HateCLIPperMMHS
from src.p2.losses import SoftBCEWithAgreementWeighting, TemperatureScaler, compute_binary_pos_weight
from src.stage1.dataset import Stage1Dataset, stage1_collate_multimodal
from src.stage1.eval import evaluate_stage1, save_stage1_results

logger = logging.getLogger(__name__)


def _device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@torch.no_grad()
def _eval_probs(model, loader, device, use_amp, amp_dtype_str):
    model.eval()
    dtype = torch.bfloat16 if amp_dtype_str == "bf16" else torch.float16
    probs, labels, ids = [], [], []
    for batch in loader:
        with autocast(enabled=use_amp and device.type == "cuda", dtype=dtype):
            logits = model(batch["image"], batch["clip_text"])
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        labels.append(batch["label_binary"].numpy())
        ids.extend(batch["tweet_id"])
    return np.concatenate(labels), np.concatenate(probs), ids


def run_hateclipper(config: HateCLIPperConfig) -> dict:
    _seed(config.seed)
    device = _device(config.device)
    logger.info(f"[HateCLIPper] device={device} fusion={config.fusion} run={config.run_name}")

    train_ds = Stage1Dataset(
        "train",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=True,
        img_size=config.img_size,
        max_samples=config.max_train_samples,
        seed=config.seed,
        use_image_store=config.use_image_store,
    )
    val_ds = Stage1Dataset(
        "val",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=True,
        img_size=config.img_size,
        max_samples=config.max_val_samples,
        seed=config.seed,
        use_image_store=config.use_image_store,
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
        collate_fn=stage1_collate_multimodal,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= config.batch_size * 2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=stage1_collate_multimodal,
        pin_memory=device.type == "cuda",
    )

    model = HateCLIPperMMHS(
        clip_model=config.clip_model,
        map_dim=config.map_dim,
        fusion=config.fusion,
        num_mapping_layers=config.num_mapping_layers,
        num_pre_output_layers=config.num_pre_output_layers,
        drop_map=config.drop_map,
        drop_fusion=config.drop_fusion,
        drop_pre=config.drop_pre,
        freeze_encoders=config.freeze_encoders,
        use_adapters=config.use_adapters,
        adapter_ratio=config.adapter_ratio,
    ).to(device)

    hard = [train_ds.labels[i]["hard_label_binary"] for i in train_ds.sample_ids]
    pos_w = compute_binary_pos_weight(hard, device)
    criterion = SoftBCEWithAgreementWeighting(
        pos_weight=pos_w,
        agreement_weights=config.agreement_weights,
        use_agreement_weighting=config.use_agreement_weighting,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    steps = max(1, len(train_loader) // config.grad_accum_steps) * config.epochs
    warmup = int(steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)

    use_amp = config.use_amp and device.type == "cuda"
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
        if config.freeze_encoders:
            model.clip.eval()
        running, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"HC ep{epoch}/{config.epochs}", leave=False)
        for step, batch in enumerate(pbar, 1):
            soft = batch["soft_hate"].to(device)
            hard_t = batch["label_binary"].to(device).float()
            targets = soft if config.use_soft_labels else hard_t
            agr = batch["agreement_level"].to(device)

            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(batch["image"], batch["clip_text"])
            loss = criterion(logits.float(), targets, agr) / config.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * config.grad_accum_steps
            n_steps = step
            pbar.set_postfix(loss=f"{running / step:.4f}")

        y_true, y_prob, ids = _eval_probs(
            model, val_loader, device, use_amp, config.amp_dtype
        )
        metrics = evaluate_stage1(y_true, y_prob)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(n_steps, 1)
        history.append(metrics)
        logger.info(
            f"[HateCLIPper] ep{epoch}: macro_f1={metrics['macro_f1']:.4f} "
            f"hate_recall={metrics['hate_recall']:.4f} auc={metrics['auc_roc']:.4f}"
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            patience = 0
            torch.save(
                {"model_state": model.state_dict(), "config": config.__dict__, "metrics": metrics},
                best_path,
            )
            save_stage1_results(config.results_run_dir, metrics, y_true, y_prob, ids)
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                logger.info("[HateCLIPper] Early stopping")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # Temperature calibration
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=use_amp, dtype=amp_dtype):
                all_logits.append(model(batch["image"], batch["clip_text"]).float().cpu())
    logits_cat = torch.cat(all_logits)
    y_true, _, ids = _eval_probs(model, val_loader, device, use_amp, config.amp_dtype)
    temp = TemperatureScaler()
    temp.fit(logits_cat, torch.tensor(y_true, dtype=torch.float32))
    y_prob = torch.sigmoid(logits_cat / temp.temperature.detach()).numpy()
    final = evaluate_stage1(y_true, y_prob)
    final["history"] = history
    final["temperature"] = float(temp.temperature.item())
    final["run_name"] = config.run_name
    save_stage1_results(config.results_run_dir, final, y_true, y_prob, ids)
    ckpt["temperature"] = final["temperature"]
    ckpt["metrics"] = final
    torch.save(ckpt, best_path)
    logger.info(f"[HateCLIPper] DONE macro_f1={final['macro_f1']:.4f}")
    return final
