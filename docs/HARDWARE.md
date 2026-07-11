# Hardware & Remote Execution Memory

## Fixed for this project

| Resource | Spec | Use |
|---|---|---|
| **Primary training GPU** | **NVIDIA RTX A6000 — 48 GB VRAM** | All Stage-1 / Stage-2 / VLM training |
| Local laptop | ~4 GB VRAM | Code edit + tiny smoke tests only (`--max_train_samples 256`) |

**Never train full MMHS models on the 4 GB local machine.** Clone this repo on the A6000 cloud box, install data via Kaggle, run `scripts/run_all_stage1.sh`.

## A6000 defaults used in configs

- AMP: `bf16` preferred (fp16 fallback)
- Batch text models: 64–128 (base), 32–64 (DeBERTa-large)
- Hate-CLIPper: batch 64, map_dim 512 (align) or 256 (cross)
- Qwen3-VL LoRA: batch 2–4, grad_accum 8–16, gradient checkpointing
- Leave ~8 GB free headroom for fragmentation

## Environment variables

```bash
export MMHS_PROJECT_ROOT=/path/to/EndSem_Project   # optional; auto-detected
export MMHS_DATA_DIR=/path/to/dataset              # if data not under project/dataset
export KAGGLE_USERNAME=...
export KAGGLE_KEY=...
```

## Dataset (Kaggle)

```
https://www.kaggle.com/datasets/ekanshkhullar/updated-hate-speech-dataset
```

Install with:

```bash
bash scripts/setup_kaggle_data.sh
# or:
python scripts/setup_kaggle_data.py
```
