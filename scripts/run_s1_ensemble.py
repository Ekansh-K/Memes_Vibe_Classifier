#!/usr/bin/env python3
"""Ensemble Stage-1 val predictions by averaging calibrated probabilities.

Looks under results/stage1/*/val_preds.npz, averages y_prob aligned by tweet_id
(or by index if ids missing), sweeps threshold, writes results/stage1/ensemble/.

  python scripts/run_s1_ensemble.py
  python scripts/run_s1_ensemble.py --runs s1_text_hate-latest_all_text s1_hateclipper_align_adapters
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stage1.eval import evaluate_stage1, save_stage1_results
from src.utils.config import PROJECT_ROOT as ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ensemble")


def load_pred(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "y_true": data["y_true"],
        "y_prob": data["y_prob"].astype(float),
    }
    if "tweet_ids" in data:
        out["tweet_ids"] = data["tweet_ids"].astype(str)
    return out


def align_and_average(preds: list[dict]) -> tuple[np.ndarray, np.ndarray, list | None]:
    if all("tweet_ids" in p for p in preds):
        # Intersection of ids
        id_sets = [set(p["tweet_ids"].tolist()) for p in preds]
        common = sorted(set.intersection(*id_sets))
        if not common:
            raise RuntimeError("No common tweet_ids across runs")
        avg = np.zeros(len(common), dtype=float)
        y_true = None
        for p in preds:
            idx = {tid: i for i, tid in enumerate(p["tweet_ids"].tolist())}
            order = [idx[t] for t in common]
            avg += p["y_prob"][order]
            if y_true is None:
                y_true = p["y_true"][order]
            else:
                if not np.array_equal(y_true, p["y_true"][order]):
                    logger.warning("y_true mismatch for some ids — using first run labels")
        avg /= len(preds)
        return y_true, avg, common

    # Index alignment
    n = min(len(p["y_prob"]) for p in preds)
    avg = np.mean([p["y_prob"][:n] for p in preds], axis=0)
    y_true = preds[0]["y_true"][:n]
    return y_true, avg, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=Path,
        default=ROOT / "results" / "stage1",
    )
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run folder names under results/stage1 (default: all with val_preds.npz)",
    )
    ap.add_argument("--min_hate_precision", type=float, default=0.40)
    args = ap.parse_args()

    results_dir = args.results_dir
    if args.runs:
        run_dirs = [results_dir / r for r in args.runs]
    else:
        run_dirs = sorted(results_dir.glob("*/val_preds.npz"))
        run_dirs = [p.parent for p in run_dirs if p.parent.name != "ensemble"]

    preds = []
    used = []
    for d in run_dirs:
        npz = d / "val_preds.npz" if d.is_dir() else d
        if not npz.exists():
            logger.warning(f"Missing {npz}")
            continue
        preds.append(load_pred(npz))
        used.append(npz.parent.name)
        logger.info(f"Loaded {npz.parent.name}: n={len(preds[-1]['y_prob'])}")

    if len(preds) < 1:
        raise SystemExit(
            f"No val_preds.npz found under {results_dir}. Train Stage-1 models first."
        )

    if len(preds) == 1:
        logger.warning("Only one model — ensemble is a no-op copy")

    y_true, y_prob, ids = align_and_average(preds)
    metrics = evaluate_stage1(y_true, y_prob, min_hate_precision=args.min_hate_precision)
    metrics["members"] = used
    metrics["n_members"] = len(used)

    out_dir = results_dir / "ensemble"
    save_stage1_results(out_dir, metrics, y_true, y_prob, ids)
    print("\n=== Ensemble Stage-1 ===")
    print(json.dumps({k: metrics[k] for k in ("macro_f1", "hate_f1", "hate_recall", "auc_roc", "threshold", "members")}, indent=2))


if __name__ == "__main__":
    main()
