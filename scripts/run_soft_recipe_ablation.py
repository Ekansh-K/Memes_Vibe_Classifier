#!/usr/bin/env python3
"""Phase 1: run soft-label recipes S0–S5 on hate-latest text (A6000).

  # Full ablation (long)
  python scripts/run_soft_recipe_ablation.py --recipes S0 S1 S2 S3 S4 S5

  # Smoke (fast)
  python scripts/run_soft_recipe_ablation.py --recipes S0 S1 S2 --max_train_samples 2000 --epochs 1

Writes results/stage1/ablation_log.csv and per-run metrics under results/stage1/.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.s1_text.config import S1TextConfig
from src.s1_text.trainer import run_s1_text
from src.stage1.soft_recipes import SOFT_RECIPES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recipes",
        nargs="+",
        default=["S0", "S1", "S2", "S3", "S4", "S5"],
        choices=list(SOFT_RECIPES),
    )
    ap.add_argument("--model", default="hate-latest")
    ap.add_argument("--text_mode", default="all_text")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_val_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--log_csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage1" / "ablation_log.csv",
    )
    args = ap.parse_args()

    summary = []
    for recipe in args.recipes:
        logger.info("=" * 60)
        logger.info(f"Starting soft recipe {recipe}")
        cfg = S1TextConfig(
            model_key=args.model,
            text_mode=args.text_mode,
            epochs=args.epochs,
            soft_recipe=recipe,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            seed=args.seed,
            device=args.device,
            early_stop_metric="macro_f1",
        )
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size

        try:
            metrics = run_s1_text(cfg)
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recipe": recipe,
                "model": args.model,
                "text_mode": args.text_mode,
                "run_name": metrics.get("run_name"),
                "macro_f1": metrics.get("macro_f1"),
                "hate_f1": metrics.get("hate_f1"),
                "hate_recall": metrics.get("hate_recall"),
                "auc_roc": metrics.get("auc_roc"),
                "threshold": metrics.get("threshold"),
                "brier_hard": metrics.get("brier_hard"),
                "brier_soft": metrics.get("brier_soft"),
                "status": "ok",
            }
        except Exception as e:
            logger.exception(f"Recipe {recipe} failed: {e}")
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "recipe": recipe,
                "model": args.model,
                "text_mode": args.text_mode,
                "run_name": "",
                "macro_f1": None,
                "hate_f1": None,
                "hate_recall": None,
                "auc_roc": None,
                "threshold": None,
                "brier_hard": None,
                "brier_soft": None,
                "status": f"fail: {e}",
            }
        append_log(args.log_csv, row)
        summary.append(row)

    print("\n=== Soft recipe ablation summary ===")
    print(f"{'recipe':6} {'macro_f1':>8} {'hate_f1':>8} {'auc':>8} {'status'}")
    for r in summary:
        mf = r["macro_f1"]
        hf = r["hate_f1"]
        au = r["auc_roc"]
        print(
            f"{r['recipe']:6} "
            f"{(mf if mf is not None else float('nan')):8.4f} "
            f"{(hf if hf is not None else float('nan')):8.4f} "
            f"{(au if au is not None else float('nan')):8.4f} "
            f"{r['status']}"
        )
    print(f"\nLog: {args.log_csv}")


if __name__ == "__main__":
    main()
