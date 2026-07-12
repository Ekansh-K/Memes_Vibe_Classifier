#!/usr/bin/env python3
"""Run Stage-1 text full fine-tune (soft recipes S0–S7).

Examples (on A6000):
  python scripts/run_s1_text.py --model hate-latest --soft_recipe S1
  python scripts/run_s1_text.py --model hate-latest --soft_recipe S5 --epochs 6
  python scripts/run_s1_text.py --model hatebert --soft_recipe S3
  python scripts/run_s1_text.py --model twitter-roberta --max_train_samples 2000  # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.s1_text.config import MODEL_PRESETS, S1TextConfig
from src.s1_text.trainer import run_s1_text
from src.stage1.soft_recipes import SOFT_RECIPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    p = argparse.ArgumentParser(description="Stage-1 text classifier (full fine-tune)")
    p.add_argument("--model", default="hate-latest", choices=list(MODEL_PRESETS.keys()))
    p.add_argument("--model_name", default="", help="Override HF model id")
    p.add_argument("--text_mode", default="all_text", choices=["tweet", "tweet_ocr", "all_text"])
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=192)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--no_soft_labels", action="store_true", help="Force hard labels (overrides recipe)")
    p.add_argument(
        "--soft_recipe",
        default="S2",
        choices=list(SOFT_RECIPES),
        help="Soft-label recipe S0–S7 (see src/stage1/soft_recipes.py)",
    )
    p.add_argument(
        "--early_stop_metric",
        default="macro_f1",
        choices=["macro_f1", "auc_roc", "hate_f1"],
    )
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--ocr_source", default="filtered")
    p.add_argument("--run_name", default="")
    args = p.parse_args()

    soft_recipe = "S0" if args.no_soft_labels else args.soft_recipe
    cfg = S1TextConfig(
        model_key=args.model,
        model_name=args.model_name or "",
        text_mode=args.text_mode,
        ocr_source=args.ocr_source,
        epochs=args.epochs,
        lr=args.lr,
        max_seq_len=args.max_seq_len,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        soft_recipe=soft_recipe,
        early_stop_metric=args.early_stop_metric,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
    )
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.run_name:
        cfg.run_name = args.run_name

    metrics = run_s1_text(cfg)
    print("\n=== Stage-1 Text Final ===")
    for k in (
        "macro_f1",
        "hate_f1",
        "hate_recall",
        "auc_roc",
        "threshold",
        "brier_hard",
        "brier_soft",
        "soft_recipe",
        "run_name",
    ):
        if k in metrics:
            print(f"  {k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
