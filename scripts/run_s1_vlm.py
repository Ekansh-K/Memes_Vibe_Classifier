#!/usr/bin/env python3
"""Run Qwen3-VL LoRA Stage-1 (not Qwen2.5).

  python scripts/run_s1_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --epochs 2
  python scripts/run_s1_vlm.py --max_train_samples 500 --epochs 1  # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.s1_vlm.config import S1VLMConfig
from src.s1_vlm.trainer import run_s1_vlm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF model id (Qwen3-VL preferred)",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--text_mode", default="all_text")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--use_4bit", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--run_name", default="")
    args = p.parse_args()

    cfg = S1VLMConfig(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lora_r=args.lora_r,
        text_mode=args.text_mode,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_4bit=args.use_4bit,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )
    if args.run_name:
        cfg.run_name = args.run_name

    metrics = run_s1_vlm(cfg)
    print("\n=== S1 VLM Final ===")
    for k in ("macro_f1", "hate_f1", "hate_recall", "auc_roc", "run_name"):
        print(f"  {k}: {metrics.get(k)}")


if __name__ == "__main__":
    main()
