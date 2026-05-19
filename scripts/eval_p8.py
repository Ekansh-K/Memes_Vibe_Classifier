"""Evaluate a trained P8 (RGCL) checkpoint on a split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import MMHS150KDataset
from src.evaluation.metrics import compute_binary_metrics, compute_multilabel_metrics
from src.models.rgcl import RGCLModel
from src.utils.config import RESULTS_DIR, DataConfig


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

    print(f"[P8-EVAL] Loaded captions for {len(captions)} samples from {path}")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--captions-file", type=str, default=None)
    parser.add_argument("--eval-threshold", type=float, default=0.5)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage = int(ckpt.get("stage", 1))
    num_classes = int(ckpt.get("num_classes", 2))
    clip_model = ckpt.get("clip_model", "ViT-L/14")
    captions_file = args.captions_file or ckpt.get("captions_file", None)
    captions = load_caption_map(captions_file)

    cfg = DataConfig()
    ds_raw = MMHS150KDataset(split=args.split, config=cfg)
    ds = build_stage_subset(ds_raw, stage)

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        tweet_ids = [str(b["tweet_id"]) for b in batch]
        ocr_texts = [b.get("ocr_text", "") or "" for b in batch]
        labels_bin = torch.tensor([int(b["label_binary"]) for b in batch], dtype=torch.long)
        labels_multi = torch.tensor([
            build_multilabel_target(ds_raw.labels[b["tweet_id"]]) for b in batch
        ], dtype=torch.float32)
        return {
            "image": images,
            "tweet_id": tweet_ids,
            "ocr_text": ocr_texts,
            "label_binary": labels_bin,
            "label_multilabel": labels_multi,
        }

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = RGCLModel(num_classes=num_classes, clip_model=clip_model)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
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
                preds = (probs >= args.eval_threshold).int().cpu().numpy().tolist()
                ys.extend(batch["label_multilabel"].cpu().numpy().tolist())
                yps.extend(preds)

    if stage == 1:
        metrics = compute_binary_metrics(ys, yps)
    else:
        metrics = compute_multilabel_metrics(ys, yps)

    print("[P8-EVAL] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_path = Path(args.output) if args.output else Path(RESULTS_DIR) / f"p8_eval_{args.split}_stage{stage}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[P8-EVAL] Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
