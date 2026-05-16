"""Transfer-training helper: load encoder weights from a pretrained checkpoint
and train for a different stage (e.g., Stage 2 hate-only 5-class) while
reinitializing the final head to match new `num_classes`.

Usage:
  python scripts/train_p7_transfer.py --pretrained checkpoints/p7_stage1_full/best.pt \
      --stage 2 --epochs 20 --batch-size 32 --lr 5e-4
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import MMHS150KDataset
from src.evaluation.metrics import compute_binary_metrics, compute_multiclass_metrics
from src.models.mhsdf import MHSDF
from src.utils.config import CHECKPOINTS_DIR, DataConfig
from src.utils.text_vectorizer import TextVectorizer


def copy_encoder_weights(pre_ckpt, model):
    src_state = pre_ckpt["model_state_dict"]
    tgt_state = model.state_dict()
    copied = 0
    for k, v in src_state.items():
        # Skip classifier head keys
        if k.startswith("head"):
            continue
        # Only copy if key exists and shape matches
        if k in tgt_state and tgt_state[k].shape == v.shape:
            tgt_state[k] = v
            copied += 1
    model.load_state_dict(tgt_state)
    print(f"[TRANSFER] Copied {copied} param tensors from pretrained checkpoint")


def collate_batch(batch, vectorizer, max_len, stage):
    import torch
    images = torch.stack([b["image"] for b in batch])
    texts = [(b.get("ocr_text", "") or "") + " " + (b.get("tweet_text", "") or "") for b in batch]
    ids_batch, lengths = vectorizer.encode(texts, max_len=max_len)
    text_ids = torch.tensor(ids_batch, dtype=torch.long)
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor([int(b["label_binary"]) if stage == 1 else max(0, int(b["label"]) - 1) for b in batch], dtype=torch.long)
    return {"image": images, "text_ids": text_ids, "lengths": lengths_t, "label": labels}


class StageCollator:
    def __init__(self, vectorizer, max_len, stage):
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.stage = stage

    def __call__(self, batch):
        return collate_batch(batch, self.vectorizer, self.max_len, self.stage)


def evaluate_model(model, loader, device, stage: int):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", "--pretrained-checkpoint", dest="pretrained", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--stage", type=int, choices=[1,2], default=2)
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience (epochs without improvement)")
    parser.add_argument("--early-stop", action="store_true", help="enable early stopping based on validation metric")
    parser.add_argument("--run-name", type=str, default="p7_transfer")
    args = parser.parse_args()

    # Load pretrained checkpoint
    pre_ckpt = torch.load(args.pretrained, map_location="cpu")
    vocab = pre_ckpt.get("vocab", None)
    if vocab is None:
        raise RuntimeError("Pretrained checkpoint does not contain 'vocab' field")

    vec = TextVectorizer(min_freq=1)
    vec.vocab = vocab

    cfg = DataConfig()
    print("[TRANSFER] Building datasets...")
    ds_train = MMHS150KDataset(split="train", config=cfg)
    ds_val = MMHS150KDataset(split="val", config=cfg)

    # Stage subset selection (hate-only for stage 2)
    def build_stage_subset_local(dataset, stage):
        if stage != 2:
            return dataset
        keep = []
        for i, sid in enumerate(dataset.split_ids):
            info = dataset.labels[sid]
            if int(info["hard_label_binary"]) == 1:
                keep.append(i)
        return Subset(dataset, keep)

    train_ds = build_stage_subset_local(ds_train, args.stage)
    val_ds = build_stage_subset_local(ds_val, args.stage)

    collate = StageCollator(vectorizer=vec, max_len=args.max_len, stage=args.stage)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    device = torch.device(args.device)
    num_classes = 2 if args.stage == 1 else 5
    model = MHSDF(vocab_size=len(vec.vocab), num_classes=num_classes, multilabel=False, freeze_cnn=False)

    # Copy encoder weights from pretrained checkpoint where possible
    copy_encoder_weights(pre_ckpt, model)

    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Checkpoints + logging
    ckpt_dir = Path(CHECKPOINTS_DIR)
    run_dir = ckpt_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "training_log.csv"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "metric_key", "metric_value"])

    print(f"[TRANSFER] Training start: device={device}, epochs={args.epochs}, stage={args.stage}, early_stop={args.early_stop}, patience={args.patience}")
    global_step = 0
    best_metric_key = "binary/macro_f1" if args.stage == 1 else "multiclass/macro_f1"
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

        train_loss = running_loss / max(n_steps, 1)
        val_metrics = evaluate_model(model, val_loader, device, stage=args.stage)

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not val_metrics:
                writer.writerow([epoch, global_step, train_loss, "", ""])
            else:
                for k, v in val_metrics.items():
                    if isinstance(v, (int, float)):
                        writer.writerow([epoch, global_step, train_loss, k, v])

        print(f"[TRANSFER] Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"  {k}: {v}")

        # Epoch checkpoint
        epoch_ckpt = run_dir / f"epoch_{epoch}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab": vec.vocab,
            "stage": args.stage,
            "num_classes": num_classes,
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
                "vocab": vec.vocab,
                "stage": args.stage,
                "num_classes": num_classes,
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_key": best_metric_key,
                "best_metric": best_metric,
            }, best_ckpt)
            print(f"[TRANSFER] New best checkpoint -> {best_ckpt} ({best_metric_key}={best_metric:.4f})")
        else:
            epochs_no_improve += 1
            print(f"[TRANSFER] No improvement for {epochs_no_improve} epoch(s)")
            if args.early_stop and epochs_no_improve >= patience:
                print(f"[TRANSFER] Early stopping triggered (no improvement in {patience} epochs).")
                break

    final_ckpt = run_dir / "transfer_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vec.vocab,
        "stage": args.stage,
        "num_classes": num_classes,
        "global_step": global_step,
    }, final_ckpt)
    print(f"[TRANSFER] Final checkpoint -> {final_ckpt}")


if __name__ == "__main__":
    main()
