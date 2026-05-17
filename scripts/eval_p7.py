"""Evaluate a trained P7 (MHSDF) checkpoint on val/test split.

Usage:
  python scripts/eval_p7.py --checkpoint checkpoints/p7_run/best.pt --split val --num-workers 0
"""

import argparse
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
from src.utils.config import CHECKPOINTS_DIR, DataConfig, RESULTS_DIR
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

    print(f"[P7-EVAL] Loaded captions for {len(captions)} samples from {path}")
    return captions


def compose_text(tweet_id: str, tweet_text: str, ocr_text: str, captions: dict[str, str], use_captions: bool) -> str:
    parts = [(ocr_text or "").strip(), (tweet_text or "").strip()]
    if use_captions:
        caption = (captions.get(str(tweet_id), "") or "").strip()
        if caption:
            parts.append(caption)
    return " ".join([p for p in parts if p]).strip()


def target_for_stage(batch_item: dict, stage: int) -> int:
    if stage == 1:
        return int(batch_item["label_binary"])
    if stage == 2:
        return max(0, int(batch_item["label"]) - 1)
    return int(batch_item["label"])


class StageCollator:
    def __init__(self, vectorizer: TextVectorizer, max_len: int, stage: int, captions: dict[str, str], use_captions: bool):
        self.vectorizer = vectorizer
        self.max_len = max_len
        self.stage = stage
        self.captions = captions
        self.use_captions = use_captions

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        texts = [
            compose_text(
                tweet_id=str(b.get("tweet_id", "")),
                tweet_text=b.get("tweet_text", "") or "",
                ocr_text=b.get("ocr_text", "") or "",
                captions=self.captions,
                use_captions=self.use_captions,
            )
            for b in batch
        ]
        ids_batch, lengths = self.vectorizer.encode(texts, max_len=self.max_len)
        text_ids = torch.tensor(ids_batch, dtype=torch.long)
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        labels = torch.tensor([target_for_stage(b, stage=self.stage) for b in batch], dtype=torch.long)
        return {"image": images, "text_ids": text_ids, "lengths": lengths_t, "label": labels}


def build_stage_subset(dataset: MMHS150KDataset, stage: int, max_samples=None):
    if stage != 2:
        if max_samples:
            return Subset(dataset, list(range(min(max_samples, len(dataset)))))
        return dataset

    keep = []
    for i, sid in enumerate(dataset.split_ids):
        info = dataset.labels[sid]
        if int(info["hard_label_binary"]) == 1:
            keep.append(i)
    if max_samples:
        keep = keep[:max_samples]
    return Subset(dataset, keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--use-captions", action="store_true", help="append VLM captions to text input")
    parser.add_argument("--captions-file", type=str, default=None, help="JSON path mapping tweet_id -> caption or {caption: ...}")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    stage = int(ckpt.get("stage", 0))
    num_classes = int(ckpt.get("num_classes", 6))
    vocab = ckpt["vocab"]
    ckpt_use_captions = bool(ckpt.get("use_captions", False))
    ckpt_captions_file = ckpt.get("captions_file", None)

    use_captions = bool(args.use_captions or ckpt_use_captions)
    captions_file = args.captions_file or ckpt_captions_file
    if use_captions and not captions_file:
        raise ValueError("Caption mode enabled but no captions file available. Pass --captions-file.")
    captions = load_caption_map(captions_file) if use_captions else {}

    vec = TextVectorizer(min_freq=1)
    vec.vocab = vocab

    cfg = DataConfig()
    ds_raw = MMHS150KDataset(split=args.split, config=cfg)
    ds = build_stage_subset(ds_raw, stage=stage, max_samples=args.max_samples)
    collate = StageCollator(vectorizer=vec, max_len=args.max_len, stage=stage, captions=captions, use_captions=use_captions)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    device = torch.device(args.device)
    model = MHSDF(vocab_size=len(vocab), num_classes=num_classes, multilabel=False, freeze_cnn=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
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
        metrics = compute_binary_metrics(ys, yps)
    else:
        metrics = compute_multiclass_metrics(ys, yps)

    print("[P7-EVAL] Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    out_path = Path(args.output) if args.output else Path(RESULTS_DIR) / f"p7_eval_{args.split}_stage{stage}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[P7-EVAL] Saved metrics -> {out_path}")


if __name__ == "__main__":
    main()
