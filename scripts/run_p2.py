"""CLI entry point for P2 (TCAM) pipeline.

Usage:
    python -m scripts.run_p2 --variation D --text_mode all_text
    python -m scripts.run_p2 --variation A --text_mode tweet_ocr
    python -m scripts.run_p2 --variation D --text_mode all_text --max_train_samples 1000
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.p2.config import P2Config
from src.p2.trainer import run_p2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_p2")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run P2 (TCAM) pipeline for MMHS-150K hate speech classification."
    )
    parser.add_argument(
        "--variation", type=str, default="D", choices=["A", "B", "C", "D"],
        help="Pipeline variation: A=binary, B=6-class, C=two-stage single-label, D=two-stage multi-label"
    )
    parser.add_argument(
        "--text_mode", type=str, default="all_text", choices=["tweet_ocr", "all_text"],
        help="Text input mode: tweet_ocr (tweet+OCR) or all_text (caption+OCR+tweet)"
    )
    parser.add_argument("--s1_epochs", type=int, default=None, help="Stage 1 epochs (overrides config)")
    parser.add_argument("--s2_epochs", type=int, default=None, help="Stage 2 epochs (overrides config)")
    parser.add_argument("--s1_lr", type=float, default=None, help="Stage 1 learning rate")
    parser.add_argument("--s2_lr", type=float, default=None, help="Stage 2 learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides both stages)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Subsample training set (for smoke tests)"
    )
    parser.add_argument(
        "--max_val_samples", type=int, default=None,
        help="Subsample validation set"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu'"
    )
    parser.add_argument(
        "--no_data_parallel", action="store_true",
        help="Disable DataParallel (single GPU mode)"
    )
    parser.add_argument(
        "--no_amp", action="store_true",
        help="Disable automatic mixed precision (fp16)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = P2Config(
        variation=args.variation,
        text_mode=args.text_mode,
        seed=args.seed,
        device=args.device,
        use_data_parallel=not args.no_data_parallel,
        use_amp=not args.no_amp,
    )
    config.run_name = f"p2_{args.variation}_{args.text_mode}"

    # Apply CLI overrides
    if args.s1_epochs is not None:
        config.s1_epochs = args.s1_epochs
    if args.s2_epochs is not None:
        config.s2_epochs = args.s2_epochs
    if args.s1_lr is not None:
        config.s1_lr = args.s1_lr
    if args.s2_lr is not None:
        config.s2_lr = args.s2_lr
    if args.batch_size is not None:
        config.s1_batch_size = args.batch_size
        config.s2_batch_size = args.batch_size
    if args.grad_accum is not None:
        config.grad_accum_steps = args.grad_accum
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_val_samples is not None:
        config.max_val_samples = args.max_val_samples

    logger.info(f"[run_p2] Config: variation={config.variation}  text_mode={config.text_mode}")
    logger.info(f"[run_p2] S1: epochs={config.s1_epochs}  lr={config.s1_lr}  batch={config.s1_batch_size}")
    logger.info(f"[run_p2] S2: epochs={config.s2_epochs}  lr={config.s2_lr}  batch={config.s2_batch_size}")
    logger.info(f"[run_p2] grad_accum={config.grad_accum_steps}  (eff_batch={config.s1_batch_size * config.grad_accum_steps})")

    metrics = run_p2(config)

    # Print summary
    print("\n" + "=" * 60)
    print(f"P2-{config.variation} [{config.text_mode}] — Results Summary")
    print("=" * 60)
    if "stage1" in metrics:
        s1_f1 = metrics["stage1"].get("best", 0.0)
        print(f"  Stage 1 best Macro F1:  {s1_f1:.4f}")
    if "stage2" in metrics:
        s2_f1 = metrics["stage2"].get("best", 0.0)
        print(f"  Stage 2 best Macro F1:  {s2_f1:.4f}")
    if "stage2_calibrated" in metrics:
        cal_f1 = metrics["stage2_calibrated"].get("multilabel/macro_f1", 0.0)
        print(f"  Stage 2 calibrated Macro F1: {cal_f1:.4f}")
    if "pipeline" in metrics:
        comp = metrics["pipeline"].get("pipeline/composite", 0.0)
        print(f"  Pipeline composite (S1_recall × S2_macro_f1): {comp:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
