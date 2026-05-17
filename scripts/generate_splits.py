#!/usr/bin/env python3
"""
Generate stratified 80/10/10 train/val/test splits for MMHS150K.

Replaces the original skewed splits (90/3.3/6.7) which had very different
hate-class distributions across splits (train=15.3% hate vs val=34.7% hate).

New splits preserve the true dataset hate rate (~17.2%) across all three
subsets, giving a valid evaluation setup.

Run:  python scripts/generate_splits.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
GT_FILE    = ROOT / "dataset" / "MMHS150K_GT.json"
LABELS_FILE = ROOT / "dataset" / "processed_labels.json"
SPLITS_DIR  = ROOT / "dataset" / "splits"

SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# TEST_FRAC  = remaining (0.10)

SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load IDs + labels ─────────────────────────────────────────────────────────
print("Loading labels …")
with open(LABELS_FILE, "r", encoding="utf-8") as f:
    labels = json.load(f)

all_ids = list(labels.keys())
print(f"Total samples with labels: {len(all_ids):,}")

# Stratify by binary label (0=NotHate, 1=Hate)
buckets: dict[int, list[str]] = defaultdict(list)
for tid in all_ids:
    cls = labels[tid]["hard_label_binary"]
    buckets[cls].append(tid)

print(f"  NotHate (0): {len(buckets[0]):,}  ({len(buckets[0])/len(all_ids)*100:.1f}%)")
print(f"  Hate    (1): {len(buckets[1]):,}  ({len(buckets[1])/len(all_ids)*100:.1f}%)")

# ── Stratified split ──────────────────────────────────────────────────────────
rng = random.Random(SEED)

train_ids, val_ids, test_ids = [], [], []

for cls, ids in sorted(buckets.items()):
    rng.shuffle(ids)
    n       = len(ids)
    n_train = round(n * TRAIN_FRAC)
    n_val   = round(n * VAL_FRAC)
    # test gets the remainder
    train_ids.extend(ids[:n_train])
    val_ids.extend(ids[n_train : n_train + n_val])
    test_ids.extend(ids[n_train + n_val :])

# Shuffle each split so they're not class-ordered
rng.shuffle(train_ids)
rng.shuffle(val_ids)
rng.shuffle(test_ids)

total = len(train_ids) + len(val_ids) + len(test_ids)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\nNew split summary (seed={SEED}):")
print(f"  Total: {total:,}")
for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    hate = sum(1 for i in ids if labels[i]["hard_label_binary"] == 1)
    pct  = len(ids) / total * 100
    print(
        f"  {name:<5}: {len(ids):>7,}  ({pct:.1f}%)  "
        f"hate={hate:,} ({hate/len(ids)*100:.1f}%)"
    )

# ── Sanity checks ─────────────────────────────────────────────────────────────
overlap_tv = set(train_ids) & set(val_ids)
overlap_tt = set(train_ids) & set(test_ids)
overlap_vt = set(val_ids) & set(test_ids)
assert len(overlap_tv) == 0, f"Train/Val overlap: {len(overlap_tv)}"
assert len(overlap_tt) == 0, f"Train/Test overlap: {len(overlap_tt)}"
assert len(overlap_vt) == 0, f"Val/Test overlap: {len(overlap_vt)}"
print("  OK  No overlap between splits")
assert total == len(all_ids), f"Total mismatch: {total} vs {len(all_ids)}"
print("  OK  All samples accounted for")

# ── Write files ───────────────────────────────────────────────────────────────
for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    out = SPLITS_DIR / f"{name}_ids.txt"
    out.write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"  Wrote: {out}  ({len(ids):,} lines)")

print("\nOK Splits written. Update TRAIN_SIZE/VAL_SIZE/TEST_SIZE in src/utils/config.py if needed.")
print(f"   TRAIN_SIZE = {len(train_ids)}")
print(f"   VAL_SIZE   = {len(val_ids)}")
print(f"   TEST_SIZE  = {len(test_ids)}")
print(f"   TOTAL_SAMPLES = {total}")
