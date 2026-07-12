#!/usr/bin/env python3
"""Run Hate-CLIPper Stage-1 on MMHS (align fusion + soft recipes + optional unfreeze).

  python scripts/run_s1_hateclipper.py --fusion align --soft_recipe S3
  python scripts/run_s1_hateclipper.py --fusion align --soft_recipe S5 --unfreeze_last_n 2
  python scripts/run_s1_hateclipper.py --fusion align --max_train_samples 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.hateclipper_mmhs.config import HateCLIPperConfig
from src.hateclipper_mmhs.trainer import run_hateclipper
from src.stage1.soft_recipes import SOFT_RECIPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fusion", default="align", choices=["align", "concat", "cross"])
    p.add_argument(
        "--no_adapters",
        action="store_true",
        help="Disable MemeCLIP-style residual adapters (enabled by default)",
    )
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--map_dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--text_mode", default="all_text", choices=["tweet", "tweet_ocr", "all_text"])
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_image_store", action="store_true")
    p.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument(
        "--soft_recipe",
        default="S2",
        choices=list(SOFT_RECIPES),
        help="Soft-label recipe S0–S7",
    )
    p.add_argument(
        "--unfreeze_last_n",
        type=int,
        default=0,
        help="Unfreeze last N CLIP vision+text blocks (0=keep frozen)",
    )
    p.add_argument("--backbone_lr", type=float, default=1e-6)
    p.add_argument(
        "--early_stop_metric",
        default="macro_f1",
        choices=["macro_f1", "auc_roc", "hate_f1"],
    )
    p.add_argument("--run_name", default="")
    args = p.parse_args()

    cfg = HateCLIPperConfig(
        fusion=args.fusion,
        use_adapters=not args.no_adapters,
        clip_model=args.clip_model,
        map_dim=args.map_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        text_mode=args.text_mode,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        use_image_store=not args.no_image_store,
        amp_dtype=args.amp_dtype,
        soft_recipe=args.soft_recipe,
        unfreeze_last_n=args.unfreeze_last_n,
        backbone_lr=args.backbone_lr,
        early_stop_metric=args.early_stop_metric,
    )
    if args.run_name:
        cfg.run_name = args.run_name

    metrics = run_hateclipper(cfg)
    print("\n=== Hate-CLIPper Final ===")
    for k in (
        "macro_f1",
        "hate_f1",
        "hate_recall",
        "auc_roc",
        "threshold",
        "brier_hard",
        "soft_recipe",
        "run_name",
    ):
        if k in metrics:
            print(f"  {k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
