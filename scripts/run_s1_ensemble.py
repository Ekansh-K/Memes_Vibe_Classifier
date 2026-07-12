#!/usr/bin/env python3
"""Ensemble Stage-1 val predictions with smart member selection + weighting.

Improvements over plain average:
  - Drop weak members (e.g. VLM with low AUC / macro F1)
  - Optional AUC- or F1-based weights
  - Brute-force best non-empty subset (up to max_subset_search members)
  - Compare mean / weighted / rank-average
  - Exclude runs by name pattern (default: drop pure-smoke-weak VLM unless strong)

  python scripts/run_s1_ensemble.py
  python scripts/run_s1_ensemble.py --exclude qwen3-vl --method auto
  python scripts/run_s1_ensemble.py --runs s1_text_hate-latest_all_text s1_hateclipper_align_adapters
"""

from __future__ import annotations

import argparse
import itertools
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


def load_member_metrics(run_dir: Path) -> dict:
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return {}
    with open(mp, "r", encoding="utf-8") as f:
        return json.load(f)


def align_probs(
    preds: list[dict],
) -> tuple[np.ndarray, list[np.ndarray], list | None]:
    """Return y_true, list of aligned y_prob arrays, optional common ids."""
    if all("tweet_ids" in p for p in preds):
        id_sets = [set(p["tweet_ids"].tolist()) for p in preds]
        common = sorted(set.intersection(*id_sets))
        if not common:
            raise RuntimeError("No common tweet_ids across runs")
        probs = []
        y_true = None
        for p in preds:
            idx = {tid: i for i, tid in enumerate(p["tweet_ids"].tolist())}
            order = [idx[t] for t in common]
            probs.append(p["y_prob"][order].astype(float))
            if y_true is None:
                y_true = p["y_true"][order]
            else:
                if not np.array_equal(y_true, p["y_true"][order]):
                    logger.warning("y_true mismatch for some ids — using first run labels")
        return y_true, probs, common

    n = min(len(p["y_prob"]) for p in preds)
    y_true = preds[0]["y_true"][:n]
    probs = [p["y_prob"][:n].astype(float) for p in preds]
    return y_true, probs, None


def combine(
    probs: list[np.ndarray],
    method: str = "mean",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    stacked = np.stack(probs, axis=0)  # (M, N)
    if method == "mean":
        return stacked.mean(axis=0)
    if method == "weighted":
        if weights is None:
            weights = np.ones(len(probs), dtype=float)
        w = np.asarray(weights, dtype=float)
        w = w / w.sum().clip(min=1e-9)
        return (stacked * w[:, None]).sum(axis=0)
    if method == "rank":
        # Average ranks then map back to [0,1] via rank/(n)
        ranks = np.zeros_like(stacked)
        for i in range(stacked.shape[0]):
            order = np.argsort(stacked[i])
            ranks[i, order] = np.linspace(0.0, 1.0, stacked.shape[1])
        return ranks.mean(axis=0)
    if method == "max":
        return stacked.max(axis=0)
    raise ValueError(f"Unknown method={method}")


def score_combo(
    y_true: np.ndarray,
    probs: list[np.ndarray],
    idxs: tuple[int, ...],
    method: str,
    member_weights: np.ndarray,
    min_hate_precision: float,
) -> tuple[float, dict, np.ndarray]:
    sub = [probs[i] for i in idxs]
    w = member_weights[list(idxs)]
    y_prob = combine(sub, method=method if method != "auto" else "weighted", weights=w)
    # For "auto" we evaluate weighted; caller tries multiple methods
    metrics = evaluate_stage1(y_true, y_prob, min_hate_precision=min_hate_precision)
    # Primary: macro_f1, tie-break AUC
    key = (metrics["macro_f1"], metrics["auc_roc"])
    return key[0] + 1e-4 * key[1], metrics, y_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=ROOT / "results" / "stage1")
    ap.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run folder names under results/stage1 (default: all with val_preds.npz)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Substring filters to drop members (e.g. qwen3-vl). "
        "Default: auto-drop members with auc < min_member_auc",
    )
    ap.add_argument(
        "--min_member_auc",
        type=float,
        default=0.52,
        help="Drop members whose solo AUC is below this "
        "(0.52 keeps modest VLMs that can still help as complementary signal)",
    )
    ap.add_argument(
        "--min_member_macro_f1",
        type=float,
        default=0.45,
        help="Drop members whose solo macro F1 is below this",
    )
    ap.add_argument(
        "--method",
        default="auto",
        choices=["auto", "mean", "weighted", "rank", "max"],
        help="auto = search mean/weighted/rank over best subset",
    )
    ap.add_argument(
        "--weight_by",
        default="auc",
        choices=["auc", "macro_f1", "hate_f1", "uniform"],
    )
    ap.add_argument(
        "--search_subsets",
        action="store_true",
        default=True,
        help="Search all non-empty subsets (default on for <=6 members)",
    )
    ap.add_argument("--no_search_subsets", action="store_true")
    ap.add_argument("--min_hate_precision", type=float, default=0.40)
    ap.add_argument(
        "--out_name",
        default="ensemble",
        help="Output folder under results/stage1 (default ensemble)",
    )
    ap.add_argument("--keep_vlm", action="store_true", help="Do not auto-exclude VLM by AUC")
    ap.add_argument(
        "--keep_short_val",
        action="store_true",
        help="Keep members with fewer val preds (e.g. 1500 vs 5000); "
        "ensemble eval then uses id intersection only",
    )
    args = ap.parse_args()

    results_dir = args.results_dir
    if args.runs:
        run_dirs = [results_dir / r for r in args.runs]
    else:
        run_dirs = sorted(
            p.parent
            for p in results_dir.glob("*/val_preds.npz")
            if p.parent.name not in ("ensemble", args.out_name)
            and not p.parent.name.startswith("ensemble")
        )

    exclude_subs = list(args.exclude or [])
    if not args.keep_vlm and not exclude_subs:
        # Soft default: name-based VLM exclude only if metrics also weak (handled below)
        pass

    candidates: list[tuple[str, Path, dict, dict]] = []
    for d in run_dirs:
        npz = d / "val_preds.npz" if d.is_dir() else d
        if not npz.exists():
            logger.warning(f"Missing {npz}")
            continue
        name = npz.parent.name
        if any(s.lower() in name.lower() for s in exclude_subs):
            logger.info(f"Excluded by --exclude: {name}")
            continue
        pred = load_pred(npz)
        meta = load_member_metrics(npz.parent)
        auc = float(meta.get("auc_roc") or 0.0)
        mf1 = float(meta.get("macro_f1") or 0.0)
        # Auto-drop weak members (typically capped VLM)
        if not args.keep_vlm:
            if auc < args.min_member_auc or mf1 < args.min_member_macro_f1:
                logger.info(
                    f"Excluded weak member {name}: macro_f1={mf1:.4f} auc={auc:.4f} "
                    f"(thresholds F1>={args.min_member_macro_f1}, AUC>={args.min_member_auc})"
                )
                continue
        candidates.append((name, npz, pred, meta))
        logger.info(
            f"Candidate {name}: n={len(pred['y_prob'])} macro_f1={mf1:.4f} auc={auc:.4f}"
        )

    # Drop short val runs (e.g. VLM on 1500) so intersection stays full-size
    # unless --keep_short_val. Intersection of 1500∩5000 silently shrinks eval set.
    if candidates and not getattr(args, "keep_short_val", False):
        max_n = max(len(c[2]["y_prob"]) for c in candidates)
        kept = []
        for name, npz, pred, meta in candidates:
            n = len(pred["y_prob"])
            if n < 0.9 * max_n:
                logger.info(
                    f"Excluded short-val member {name}: n={n} (full val n≈{max_n}). "
                    "Use --keep_short_val to force include (shrinks eval to intersection)."
                )
                continue
            kept.append((name, npz, pred, meta))
        if kept:
            candidates = kept
        else:
            logger.warning("All members short? Keeping originals.")

    if not candidates:
        raise SystemExit(
            f"No val_preds.npz found under {results_dir} after filtering. "
            "Train Stage-1 models first or loosen --min_member_auc."
        )

    names = [c[0] for c in candidates]
    preds = [c[2] for c in candidates]
    metas = [c[3] for c in candidates]

    y_true, probs, ids = align_probs(preds)

    # Member weights
    def weight_vec(metas_list: list[dict]) -> np.ndarray:
        if args.weight_by == "uniform":
            return np.ones(len(metas_list), dtype=float)
        key = {
            "auc": "auc_roc",
            "macro_f1": "macro_f1",
            "hate_f1": "hate_f1",
        }[args.weight_by]
        w = np.array([max(float(m.get(key) or 0.0), 1e-3) for m in metas_list], dtype=float)
        # Square soft-max emphasis on stronger members
        w = w ** 2
        return w

    methods = (
        ["mean", "weighted", "rank"]
        if args.method == "auto"
        else [args.method]
    )
    search = args.search_subsets and not args.no_search_subsets
    n = len(names)
    if n > 8:
        search = False
        logger.warning("Too many members for subset search — using all members only")

    best = {
        "score": -1.0,
        "metrics": None,
        "y_prob": None,
        "members": None,
        "method": None,
        "weights": None,
    }

    index_sets: list[tuple[int, ...]]
    if search:
        index_sets = []
        for k in range(1, n + 1):
            index_sets.extend(itertools.combinations(range(n), k))
    else:
        index_sets = [tuple(range(n))]

    # Quiet eval logging during combinatorial search
    eval_logger = logging.getLogger("src.stage1.eval")
    prev_level = eval_logger.level
    eval_logger.setLevel(logging.WARNING)
    for idxs in index_sets:
        w_all = weight_vec([metas[i] for i in idxs])
        for method in methods:
            sub_probs = [probs[i] for i in idxs]
            y_prob = combine(sub_probs, method=method, weights=w_all)
            metrics = evaluate_stage1(
                y_true, y_prob, min_hate_precision=args.min_hate_precision
            )
            # Prefer higher macro F1 / AUC; small bonus for multi-member diversity
            n_mem = len(idxs)
            score = (
                metrics["macro_f1"]
                + 1e-4 * metrics["auc_roc"]
                + 1e-5 * max(n_mem - 1, 0)
            )
            # Skip degenerate "ensemble" of 1 using rank (rank remap alone is not ensembling)
            if n_mem == 1 and method == "rank":
                continue
            if score > best["score"]:
                best.update(
                    score=score,
                    metrics=metrics,
                    y_prob=y_prob,
                    members=[names[i] for i in idxs],
                    method=method,
                    weights={
                        names[i]: float(w_all[j]) for j, i in enumerate(idxs)
                    },
                )
    eval_logger.setLevel(prev_level)
    if best["metrics"] is None:
        # Fallback: simple mean of all candidates
        y_prob = combine(probs, method="mean")
        metrics = evaluate_stage1(
            y_true, y_prob, min_hate_precision=args.min_hate_precision
        )
        best.update(
            score=metrics["macro_f1"],
            metrics=metrics,
            y_prob=y_prob,
            members=names,
            method="mean",
            weights={n: 1.0 for n in names},
        )

    assert best["metrics"] is not None
    metrics = best["metrics"]
    metrics["members"] = best["members"]
    metrics["n_members"] = len(best["members"])
    metrics["ensemble_method"] = best["method"]
    metrics["ensemble_weights"] = best["weights"]
    metrics["weight_by"] = args.weight_by
    metrics["excluded_policy"] = {
        "min_member_auc": args.min_member_auc,
        "min_member_macro_f1": args.min_member_macro_f1,
        "exclude": exclude_subs,
    }

    # Also report all-mean of strong members for reference
    strong_mean = combine(probs, method="mean")
    ref = evaluate_stage1(y_true, strong_mean, min_hate_precision=args.min_hate_precision)
    metrics["ref_all_strong_mean"] = {
        "macro_f1": ref["macro_f1"],
        "auc_roc": ref["auc_roc"],
        "members": names,
    }

    out_dir = results_dir / args.out_name
    save_stage1_results(out_dir, metrics, y_true, best["y_prob"], ids)

    print("\n=== Ensemble Stage-1 (improved) ===")
    print(
        json.dumps(
            {
                "macro_f1": metrics["macro_f1"],
                "hate_f1": metrics["hate_f1"],
                "hate_recall": metrics["hate_recall"],
                "auc_roc": metrics["auc_roc"],
                "threshold": metrics["threshold"],
                "method": best["method"],
                "members": best["members"],
                "weights": best["weights"],
                "ref_all_strong_mean": metrics["ref_all_strong_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
