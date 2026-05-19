"""Train script for P8 (RGCL) — CLIP + retrieval-guided contrastive learning.

Two-stage:
  - Stage 1: binary (hateful vs not_hate)
  - Stage 2: 5-class multi-label (hateful-only subset)

Uses soft labels for training and early stopping on validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.data.dataset import MMHS150KDataset
from src.evaluation.metrics import compute_binary_metrics, compute_multilabel_metrics
from src.models.rgcl import RGCLModel
from src.utils.config import CHECKPOINTS_DIR, DataConfig

try:
    import faiss
except Exception:
    faiss = None


def load_caption_map(captions_file: str | None) -> dict[str, str]:
    if not captions_file:
        return {}
    path = Path(captions_file)
    if not path.exists():
        raise FileNotFoundError(f"Captions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    captions = {}
    for tweet_id, value in raw.items():
        if isinstance(value, dict):
            caption = (value.get("caption", "") or "").strip()
        elif isinstance(value, str):
            caption = value.strip()
        else:
            caption = ""
        if caption:
            captions[str(tweet_id)] = caption

    print(f"[P8] Loaded captions for {len(captions)} samples from {path}")
    return captions


def compose_text(tweet_id: str, ocr_text: str, captions: dict[str, str]) -> str:
    caption = (captions.get(str(tweet_id), "") or "").strip()
    ocr = (ocr_text or "").strip()
    if caption and ocr:
        return f"{caption} [SEP] {ocr}"
    return caption or ocr


def build_multilabel_target(label_info: dict, num_classes: int = 5, threshold: int = 2) -> list[int]:
    votes = label_info.get("annotator_labels", [])
    counts = [0] * (num_classes + 1)
    for v in votes:
        if 0 <= int(v) <= num_classes:
            counts[int(v)] += 1
    targets = []
    for cls in range(1, num_classes + 1):
        targets.append(1 if counts[cls] >= threshold else 0)
    return targets


def build_stage_subset(dataset: MMHS150KDataset, stage: int):
    if stage != 2:
        return dataset
    keep = []
    for i, sid in enumerate(dataset.split_ids):
        info = dataset.labels[sid]
        if int(info["hard_label_binary"]) == 1:
            keep.append(i)
    return Subset(dataset, keep)


def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    neighbor_embeddings: torch.Tensor,
    neighbor_labels: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    bsz, k, _ = neighbor_embeddings.shape
    losses = []
    for i in range(bsz):
        q = embeddings[i]
        sims = F.cosine_similarity(q.unsqueeze(0).expand(k, -1), neighbor_embeddings[i]) / tau
        if labels.dim() == 1:
            pos_mask = (neighbor_labels[i] == labels[i]).float()
        else:
            shared = (neighbor_labels[i] * labels[i].unsqueeze(0)).sum(dim=-1)
            pos_mask = (shared > 0).float()
        if pos_mask.sum() == 0:
            continue
        log_sum_exp = torch.logsumexp(sims, dim=0)
        pos_sim = (sims * pos_mask).sum() / pos_mask.sum()
        losses.append(log_sum_exp - pos_sim)
    if not losses:
        return torch.tensor(0.0, device=embeddings.device)
    return torch.stack(losses).mean()


def build_faiss_index(model, loader, device, stage: int, captions: dict[str, str]):
    if faiss is None:
        raise RuntimeError("faiss is not available; install faiss-cpu")
    model.eval()
    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            texts = [
                compose_text(bid, ocr, captions)
                for bid, ocr in zip(batch["tweet_id"], batch["ocr_text"])
            ]
            _, emb = model(images, texts)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb.cpu().numpy())
            if stage == 1:
                all_labels.append(batch["label_binary"].cpu().numpy())
            else:
                all_labels.append(batch["label_multilabel"].cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    all_labels = np.concatenate(all_labels, axis=0)

    index = faiss.IndexFlatIP(all_embs.shape[1])
    faiss.normalize_L2(all_embs)
    index.add(all_embs)
    return index, all_embs, all_labels


def evaluate_model(model, loader, device, stage: int, captions: dict[str, str], threshold: float):
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            texts = [
                compose_text(bid, ocr, captions)
                for bid, ocr in zip(batch["tweet_id"], batch["ocr_text"])
            ]
            logits, _ = model(images, texts)
            if stage == 1:
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1).cpu().numpy().tolist()
                ys.extend(batch["label_binary"].cpu().numpy().tolist())
                yps.extend(preds)
            else:
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).int().cpu().numpy().tolist()
                ys.extend(batch["label_multilabel"].cpu().numpy().tolist())
                yps.extend(preds)

    if stage == 1:
        return compute_binary_metrics(ys, yps)
    return compute_multilabel_metrics(ys, yps)


def ensure_csv_header(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "train_loss", "metric_key", "metric_value"])


def append_csv_rows(path: Path, epoch: int, step: int, train_loss: float, metrics: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not metrics:
            writer.writerow([epoch, step, train_loss, "", ""])
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                writer.writerow([epoch, step, train_loss, k, v])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="p8_run")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--contrastive-weight", type=float, default=0.5)
    parser.add_argument("--contrastive-tau", type=float, default=0.07)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--rebuild-index-every", type=int, default=2)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    parser.add_argument("--captions-file", type=str, required=True)
    args = parser.parse_args()

    if faiss is None:
        raise RuntimeError("faiss is not available; install faiss-cpu")

    captions = load_caption_map(args.captions_file)
    cfg = DataConfig()

    print("[P8] Building datasets...")
    ds_train = MMHS150KDataset(split="train", config=cfg)
    ds_val = MMHS150KDataset(split="val", config=cfg)

    train_ds = build_stage_subset(ds_train, args.stage)
    val_ds = build_stage_subset(ds_val, args.stage)

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        tweet_ids = [str(b["tweet_id"]) for b in batch]
        ocr_texts = [b.get("ocr_text", "") or "" for b in batch]
        labels_bin = torch.tensor([int(b["label_binary"]) for b in batch], dtype=torch.long)
        soft_bin = torch.tensor([
            ds_train.labels[b["tweet_id"]]["soft_label_binary"] for b in batch
        ], dtype=torch.float32)
        label_multi = torch.tensor([
            build_multilabel_target(ds_train.labels[b["tweet_id"]]) for b in batch
        ], dtype=torch.float32)
        soft_multi = torch.tensor([
            ds_train.labels[b["tweet_id"]]["soft_label_6class"][1:] for b in batch
        ], dtype=torch.float32)
        return {
            "image": images,
            "tweet_id": tweet_ids,
            "ocr_text": ocr_texts,
            "label_binary": labels_bin,
            "soft_binary": soft_bin,
            "label_multilabel": label_multi,
            "soft_multilabel": soft_multi,
        }

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    num_classes = 2 if args.stage == 1 else 5
    model = RGCLModel(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    ckpt_dir = Path(CHECKPOINTS_DIR)
    run_dir = ckpt_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"
    ensure_csv_header(csv_path)

    print(
        f"[P8] Training start: device={device}, stage={args.stage}, epochs={args.epochs}, early_stop={args.early_stop}"
    )

    best_metric_key = "binary/macro_f1" if args.stage == 1 else "multilabel/macro_f1"
    best_metric = -1.0
    epochs_no_improve = 0

    index_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    faiss_index, train_embs, train_labels = build_faiss_index(model, index_loader, device, args.stage, captions)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or (args.rebuild_index_every > 0 and epoch % args.rebuild_index_every == 0):
            faiss_index, train_embs, train_labels = build_faiss_index(model, index_loader, device, args.stage, captions)
            print(f"[P8] Rebuilt FAISS index at epoch {epoch}")

        model.train()
        running_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            global_step += 1
            images = batch["image"].to(device)
            texts = [
                compose_text(bid, ocr, captions)
                for bid, ocr in zip(batch["tweet_id"], batch["ocr_text"])
            ]

            logits, emb = model(images, texts)

            if args.stage == 1:
                soft_targets = batch["soft_binary"].to(device)
                cls_loss = soft_ce_loss(logits, soft_targets)
                label_for_retrieval = batch["label_binary"].to(device)
            else:
                soft_targets = batch["soft_multilabel"].to(device)
                cls_loss = F.binary_cross_entropy_with_logits(logits, soft_targets)
                label_for_retrieval = batch["label_multilabel"].to(device)

            with torch.no_grad():
                emb_norm = emb / emb.norm(dim=-1, keepdim=True)
                emb_np = emb_norm.detach().cpu().numpy().astype(np.float32)
                _, nn_indices = faiss_index.search(emb_np, args.top_k + 1)

            nn_embs = torch.tensor(train_embs[nn_indices[:, 1:]], dtype=torch.float32, device=device)
            nn_labels = torch.tensor(train_labels[nn_indices[:, 1:]], dtype=torch.float32, device=device)
            contr_loss = contrastive_loss(emb, label_for_retrieval, nn_embs, nn_labels, args.contrastive_tau)

            loss = cls_loss + args.contrastive_weight * contr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1

        train_loss = running_loss / max(n_steps, 1)
        val_metrics = evaluate_model(model, val_loader, device, args.stage, captions, args.eval_threshold)
        append_csv_rows(csv_path, epoch=epoch, step=global_step, train_loss=train_loss, metrics=val_metrics)

        print(f"[P8] Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")

        epoch_ckpt = run_dir / f"epoch_{epoch}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "stage": args.stage,
            "num_classes": num_classes,
            "clip_model": model.clip_model_name,
            "captions_file": args.captions_file,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_tau": args.contrastive_tau,
            "top_k": args.top_k,
            "epoch": epoch,
            "global_step": global_step,
        }, epoch_ckpt)

        current = float(val_metrics.get(best_metric_key, -1.0))
        if current > best_metric:
            best_metric = current
            epochs_no_improve = 0
            best_ckpt = run_dir / "best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "stage": args.stage,
                "num_classes": num_classes,
                "clip_model": model.clip_model_name,
                "captions_file": args.captions_file,
                "contrastive_weight": args.contrastive_weight,
                "contrastive_tau": args.contrastive_tau,
                "top_k": args.top_k,
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_key": best_metric_key,
                "best_metric": best_metric,
            }, best_ckpt)
            print(f"[P8] New best checkpoint -> {best_ckpt} ({best_metric_key}={best_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[P8] No improvement for {epochs_no_improve} epoch(s)")
            if args.early_stop and epochs_no_improve >= args.patience:
                print(f"[P8] Early stopping triggered (no improvement in {args.patience} epochs).")
                break

    final_ckpt = run_dir / "final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": args.stage,
        "num_classes": num_classes,
        "clip_model": model.clip_model_name,
        "captions_file": args.captions_file,
        "contrastive_weight": args.contrastive_weight,
        "contrastive_tau": args.contrastive_tau,
        "top_k": args.top_k,
        "global_step": global_step,
    }, final_ckpt)
    print(f"[P8] Final checkpoint -> {final_ckpt}")


if __name__ == "__main__":
    main()
