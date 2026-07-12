# Stage-1 Soft-Label Recipes (S0–S7)

Implementation of the accuracy uplift plan: soft labels were already on, but
`pos_weight` + aggressive agreement downweighting likely cancelled the soft signal.

## Quick start (A6000)

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD PYTHONUNBUFFERED=1

# Baseline report of existing runs
python scripts/report_stage1_baselines.py

# Soft-recipe ablation on hate-latest (full data, ~1–2h each)
python scripts/run_soft_recipe_ablation.py --recipes S0 S1 S2 S3 S4 S5 --epochs 4

# Smoke (local / quick)
python scripts/run_soft_recipe_ablation.py --recipes S0 S1 S2 \
  --max_train_samples 2000 --max_val_samples 500 --epochs 1

# Single text run with winner recipe
python scripts/run_s1_text.py --model hate-latest --soft_recipe S5 --epochs 6

# Hate-CLIPper with soft recipe + partial unfreeze
python scripts/run_s1_hateclipper.py --fusion align --soft_recipe S3 --unfreeze_last_n 2

# Ensemble after members exist
python scripts/run_s1_ensemble.py
```

## Recipe table

| ID | Targets | pos_weight | Agreement weights | Notes |
|----|---------|------------|-------------------|-------|
| **S0** | Hard | yes | off | Classic imbalanced baseline |
| **S1** | Soft | **no** | off | Soft without imbalance hammer |
| **S2** | Soft | yes | (0.4, 0.7, 1.0) | **Legacy default** |
| **S3** | Soft | no | (0.7, 0.9, 1.0) binary | Gentler weights on 2–1 samples |
| **S4** | Soft | no | off | Pure soft BCE |
| **S5** | Hard + soft multi-task | hard yes / soft no | off | Fornaciari-style |
| **S6** | Soft | no | gentle | Train filter `agreement_binary ≥ 2` |
| **S7** | Soft sharpened | no | gentle | Soft temperature &lt; 1 |

## Binary agreement

`agreement_binary` is computed from hate vs not-hate votes only (not 6-class).
Loaded labels backfill it on the fly if `processed_labels.json` is older.

To persist:
```bash
python -c "from src.data.splits import generate_processed_labels; generate_processed_labels()"
```

## Metrics always logged

- Hard Macro F1 / Hate F1 / Recall / AUC + threshold sweep
- `brier_hard`, optional `brier_soft`
- Agreement-stratified metrics (`unanimous` / `majority`)

## Selection rule

Pick recipe **W** that maximizes val Macro F1; if tie, prefer higher AUC.
Then retrain text multi-seed + Hate-CLIPper with **W**, then ensemble.

## VLM

Deferred for Stage-1 accuracy. Use `--skip-vlm` on `run_all_stage1.sh` if needed.
