#!/usr/bin/env python3
"""Phase 0: dump Stage-1 baseline metrics from results/stage1/*/metrics.json.

  python scripts/report_stage1_baselines.py
  python scripts/report_stage1_baselines.py --results_dir results/stage1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results_dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage1",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "stage1" / "baseline_report.csv",
    )
    args = ap.parse_args()

    rows = []
    if not args.results_dir.exists():
        print(f"No results dir: {args.results_dir}")
        return

    for metrics_path in sorted(args.results_dir.glob("*/metrics.json")):
        run = metrics_path.parent.name
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        cfg_path = (
            PROJECT_ROOT / "checkpoints" / "stage1" / run / "config.yaml"
        )
        soft_flags = ""
        if cfg_path.exists():
            soft_flags = cfg_path.read_text(encoding="utf-8")[:500].replace("\n", " | ")
        row = {
            "run": run,
            "macro_f1": m.get("macro_f1"),
            "hate_f1": m.get("hate_f1"),
            "hate_recall": m.get("hate_recall"),
            "auc_roc": m.get("auc_roc"),
            "threshold": m.get("threshold"),
            "brier_hard": m.get("brier_hard"),
            "brier_soft": m.get("brier_soft"),
            "soft_recipe": m.get("soft_recipe"),
            "n_samples": (m.get("at_best_macro_f1") or {}).get("n_samples"),
            "config_snippet": soft_flags[:200],
        }
        rows.append(row)

    # Sort by macro_f1 desc
    rows.sort(key=lambda r: (r["macro_f1"] is not None, r["macro_f1"] or 0), reverse=True)

    print("=" * 88)
    print(f"{'run':45} {'macro_f1':>8} {'hate_f1':>8} {'recall':>8} {'auc':>8} {'recipe':>6}")
    print("-" * 88)
    for r in rows:
        print(
            f"{r['run'][:45]:45} "
            f"{(r['macro_f1'] or 0):8.4f} "
            f"{(r['hate_f1'] or 0):8.4f} "
            f"{(r['hate_recall'] or 0):8.4f} "
            f"{(r['auc_roc'] or 0):8.4f} "
            f"{(r['soft_recipe'] or '-'):>6}"
        )
    print("=" * 88)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
