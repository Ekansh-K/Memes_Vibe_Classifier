"""Train script for P7 (MHSDF) — ResNet18 + BiLSTM baseline.

Upgrades over initial smoke script:
- periodic and per-epoch checkpoints
- CSV logging for train/val metrics
- optional W&B logging
- stage-aware training:
  - stage 1: binary (not_hate vs hate)
  - stage 2: hate-category-only (classes 1..5 mapped to 0..4)
  - default: full 6-class multiclass

Examples:
  python scripts/train_p7.py --epochs 3 --batch-size 32 --num-workers 0
  python scripts/train_p7.py --stage 1 --epochs 3 --batch-size 32 --num-workers 0
  python scripts/train_p7.py --stage 2 --epochs 3 --batch-size 32 --num-workers 0
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import MMHS150KDataset
from src.data.splits import load_gt_json, load_ocr_data, load_split_ids
from src.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics
from src.models.mhsdf import MHSDF
from src.utils.config import CHECKPOINTS_DIR, DataConfig
from src.utils.text_vectorizer import TextVectorizer


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

    print(f"[P7] Loaded captions for {len(captions)} samples from {path}")
    return captions


def compose_text(tweet_id: str, tweet_text: str, ocr_text: str, captions: dict[str, str], use_captions: bool) -> str:
    parts = [(ocr_text or "").strip(), (tweet_text or "").strip()]
    if use_captions:
        caption = (captions.get(str(tweet_id), "") or "").strip()
        if caption:
            parts.append(caption)
    return " ".join([p for p in parts if p]).strip()


def build_texts_for_vocab(max_ids=None, captions: dict[str, str] | None = None, use_captions: bool = False):
    gt = load_gt_json()
    ocr = load_ocr_data()
    ids = list(load_split_ids("train"))
    captions = captions or {}
    if max_ids:
        ids = ids[:max_ids]
    texts = []
    for tid in ids:
        entry = gt.get(tid, {})
        t = entry.get("tweet_text", "") or ""
        o = ocr.get(tid, "") or ""
        texts.append(compose_text(tid, t, o, captions=captions, use_captions=use_captions))
    return texts


def target_for_stage(batch_item: dict, stage: int) -> int:
    if stage == 1:
        return int(batch_item["label_binary"])
    if stage == 2:
        # Map hate classes 1..5 -> 0..4. If not-hate is encountered unexpectedly,
        # map to class 0 to keep shape safe.
        raw = int(batch_item["label"])
        return max(0, raw - 1)
    return int(batch_item["label"])


def collate_batch(batch, vectorizer: TextVectorizer, max_len: int, stage: int, captions: dict[str, str], use_captions: bool):
    images = torch.stack([b["image"] for b in batch])
    texts = [
        compose_text(
            tweet_id=str(b.get("tweet_id", "")),
            tweet_text=b.get("tweet_text", "") or "",
            ocr_text=b.get("ocr_text", "") or "",
            captions=captions,
            use_captions=use_captions,
        )
        for b in batch
    ]
    ids_batch, lengths = vectorizer.encode(texts, max_len=max_len)
    text_ids = torch.tensor(ids_batch, dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor([target_for_stage(b, stage=stage) for b in batch], dtype=torch.long)
    return {"image": images, "text_ids": text_ids, "lengths": lengths_t, "label": labels}


class StageCollator:
    """Pickle-safe callable collator for Windows multiprocessing."""

    def __init__(self, vectorizer: TextVectorizer, max_len: int, stage: int, captions: dict[str, str], use_captions: bool):
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.stage = stage
        self.captions = captions
        self.use_captions = use_captions

    def __call__(self, batch):
        return collate_batch(batch, self.vectorizer, self.max_len, self.stage, self.captions, self.use_captions)


def build_stage_subset(dataset: MMHS150KDataset, stage: int, max_samples=None):
    if stage != 2:
        if max_samples:
            return Subset(dataset, list(range(min(max_samples, len(dataset)))))
        return dataset

    # Stage-2 uses hateful-only examples.
    keep = []
    for i, sid in enumerate(dataset.split_ids):
        info = dataset.labels[sid]
        if int(info["hard_label_binary"]) == 1:
            keep.append(i)
    if max_samples:
        keep = keep[:max_samples]
    return Subset(dataset, keep)


def evaluate_model(model, loader, device, stage: int) -> dict:
    model.eval()
    ys = []
    yps = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].cpu().numpy().tolist()
            logits = model(images, text_ids, lengths)
            preds = logits.argmax(dim=-1).cpu().numpy().tolist()
            ys.extend(labels)
            yps.extend(preds)

    if stage == 1:
        return compute_binary_metrics(ys, yps)
    return compute_multiclass_metrics(ys, yps)


def maybe_init_wandb(args, config_dict):
    if not args.wandb:
        return None
    try:
        import wandb

        run = wandb.init(project=args.wandb_project, name=args.run_name, config=config_dict)
        return run
    except Exception as e:
        print(f"[P7] W&B disabled due to error: {e}")
        return None


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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="dataloader num_workers (set 0 on Windows)")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2], default=0, help="0=full 6-class, 1=binary, 2=hate-only 5-class")
    parser.add_argument("--save-every", type=int, default=0, help="save step checkpoint every N steps (0 disables)")
    parser.add_argument("--checkpoint-name", type=str, default="p7_baseline.pt")
    parser.add_argument("--run-name", type=str, default="p7_run")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="mmhs150k-hate-speech")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without improvement)")
    parser.add_argument("--early-stop", action="store_true", help="enable early stopping based on validation metric")
    parser.add_argument("--use-captions", action="store_true", help="append VLM captions to text input")
    parser.add_argument("--captions-file", type=str, default=None, help="JSON path mapping tweet_id -> caption or {caption: ...}")
    args = parser.parse_args()

    if args.use_captions and not args.captions_file:
        raise ValueError("--use-captions requires --captions-file")

    captions = load_caption_map(args.captions_file) if args.use_captions else {}

    cfg = DataConfig()
    stage = int(args.stage)

    print("[P7] Building vocab from train split...")
    texts = build_texts_for_vocab(max_ids=args.max_samples, captions=captions, use_captions=args.use_captions)
    vectorizer = TextVectorizer(min_freq=1)
    vectorizer.fit(texts)
    print(f"[P7] Vocab size: {vectorizer.vocab_size}")

    print("[P7] Preparing datasets and dataloaders...")
    train_ds_raw = MMHS150KDataset(split="train", config=cfg)
    val_ds_raw = MMHS150KDataset(split="val", config=cfg)

    train_ds = build_stage_subset(train_ds_raw, stage=stage, max_samples=args.max_samples)
    val_ds = build_stage_subset(val_ds_raw, stage=stage, max_samples=None)

    collate = StageCollator(
        vectorizer=vectorizer,
        max_len=args.max_len,
        stage=stage,
        captions=captions,
        use_captions=args.use_captions,
    )
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
    if stage == 1:
        num_classes = 2
    elif stage == 2:
        num_classes = 5
    else:
        num_classes = 6

    model = MHSDF(vocab_size=vectorizer.vocab_size, num_classes=num_classes, multilabel=False, freeze_cnn=True)
    model.to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    ckpt_dir = Path(CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = ckpt_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"
    ensure_csv_header(csv_path)

    cfg_for_log = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_samples": args.max_samples,
        "max_len": args.max_len,
        "stage": stage,
        "num_classes": num_classes,
        "use_captions": args.use_captions,
        "captions_file": args.captions_file,
    }
    wandb_run = maybe_init_wandb(args, cfg_for_log)

    print(
        f"[P7] Training start: device={device}, epochs={args.epochs}, stage={stage}, "
        f"early_stop={args.early_stop}, patience={args.patience}"
    )

    global_step = 0
    best_metric_key = "binary/macro_f1" if stage == 1 else "multiclass/macro_f1"
    best_metric = -1.0
    epochs_no_improve = 0
    patience = int(args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            global_step += 1
            images = batch["image"].to(device)
            text_ids = batch["text_ids"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images, text_ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_steps += 1

            if args.save_every > 0 and global_step % args.save_every == 0:
                step_ckpt = run_dir / f"step_{global_step}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab": vectorizer.vocab,
                        "stage": stage,
                        "num_classes": num_classes,
                        "use_captions": args.use_captions,
                        "captions_file": args.captions_file,
                        "global_step": global_step,
                    },
                    step_ckpt,
                )
                print(f"[P7] Saved step checkpoint -> {step_ckpt}")

        train_loss = running_loss / max(n_steps, 1)
        val_metrics = evaluate_model(model, val_loader, device, stage=stage)

        append_csv_rows(csv_path, epoch=epoch, step=global_step, train_loss=train_loss, metrics=val_metrics)

        if wandb_run is not None:
            import wandb

            wandb.log({"train/loss": train_loss, "epoch": epoch, **val_metrics}, step=global_step)

        print(f"[P7] Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")
        print("[P7] Validation metrics:")
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v}")

        # Epoch checkpoint
        epoch_ckpt = run_dir / f"epoch_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab": vectorizer.vocab,
                "stage": stage,
                "num_classes": num_classes,
                "use_captions": args.use_captions,
                "captions_file": args.captions_file,
                "epoch": epoch,
                "global_step": global_step,
            },
            epoch_ckpt,
        )

        current = float(val_metrics.get(best_metric_key, -1.0))
        if current > best_metric:
            best_metric = current
            epochs_no_improve = 0
            best_ckpt = run_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vectorizer.vocab,
                    "stage": stage,
                    "num_classes": num_classes,
                    "use_captions": args.use_captions,
                    "captions_file": args.captions_file,
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metric_key": best_metric_key,
                    "best_metric": best_metric,
                },
                best_ckpt,
            )
            print(f"[P7] New best checkpoint -> {best_ckpt} ({best_metric_key}={best_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[P7] No improvement for {epochs_no_improve} epoch(s)")
            if args.early_stop and epochs_no_improve >= patience:
                print(f"[P7] Early stopping triggered (no improvement in {patience} epochs).")
                break

    final_ckpt = run_dir / args.checkpoint_name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vectorizer.vocab,
            "stage": stage,
            "num_classes": num_classes,
            "use_captions": args.use_captions,
            "captions_file": args.captions_file,
            "global_step": global_step,
        },
        final_ckpt,
    )
    print(f"[P7] Final checkpoint -> {final_ckpt}")
    print(f"[P7] CSV log -> {csv_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
