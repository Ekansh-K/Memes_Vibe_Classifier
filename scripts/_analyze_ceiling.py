"""Deep analysis of Stage 1 ceiling: label noise, annotator disagreement,
class-conditional accuracy, and feature quality."""
import sys; sys.path.insert(0, '.')
import json
from collections import Counter
from pathlib import Path
from src.data.splits import load_processed_labels, load_split_ids, load_gt_json

labels = load_processed_labels()
gt = load_gt_json()

# ── 1. Annotator agreement analysis ──────────────────────────────────────────
print("="*60)
print("1. ANNOTATOR AGREEMENT ANALYSIS")
print("="*60)

agree_counts = Counter()
binary_agree_counts = Counter()
total = 0

# Track per-agreement-level binary distribution
agree_hate = {1: 0, 2: 0, 3: 0}
agree_nothate = {1: 0, 2: 0, 3: 0}

for tid, info in labels.items():
    al = info["agreement_level"]
    agree_counts[al] += 1
    total += 1

    ann = info["annotator_labels"]
    # Binary agreement: do annotators agree on hate vs not-hate?
    bin_labels = [0 if a == 0 else 1 for a in ann]
    if len(set(bin_labels)) == 1:
        binary_agree_counts["unanimous"] += 1
    elif bin_labels.count(1) >= 2 or bin_labels.count(0) >= 2:
        binary_agree_counts["majority"] += 1
    else:
        binary_agree_counts["split"] += 1

    if info["hard_label_binary"] == 1:
        agree_hate[al] += 1
    else:
        agree_nothate[al] += 1

print(f"\n6-class agreement:")
for level in [3, 2, 1]:
    n = agree_counts[level]
    desc = {3: "All 3 agree", 2: "2 of 3 agree", 1: "All differ"}[level]
    print(f"  Level {level} ({desc}): {n:>7,}  ({n/total*100:5.1f}%)")

print(f"\nBinary (hate/not-hate) agreement:")
for k in ["unanimous", "majority", "split"]:
    n = binary_agree_counts[k]
    print(f"  {k:<10}: {n:>7,}  ({n/total*100:5.1f}%)")

print(f"\nHate rate by agreement level:")
for level in [3, 2, 1]:
    h = agree_hate[level]
    nh = agree_nothate[level]
    t = h + nh
    print(f"  Level {level}: {h:>6,} hate / {t:>7,} total  = {h/t*100:5.1f}%")

# ── 2. Noisy label analysis ──────────────────────────────────────────────────
print("\n" + "="*60)
print("2. LABEL NOISE — SAMPLES WHERE BINARY LABEL IS AMBIGUOUS")
print("="*60)

# A sample is "ambiguous" if annotators disagree on hate vs not-hate
ambiguous = 0
ambig_labeled_hate = 0
ambig_labeled_nothate = 0
for tid, info in labels.items():
    ann = info["annotator_labels"]
    bin_labels = [0 if a == 0 else 1 for a in ann]
    if len(set(bin_labels)) > 1:  # at least one disagrees
        ambiguous += 1
        if info["hard_label_binary"] == 1:
            ambig_labeled_hate += 1
        else:
            ambig_labeled_nothate += 1

print(f"Ambiguous samples (annotators disagree on hate/not-hate): {ambiguous:,}")
print(f"  % of total: {ambiguous/total*100:.1f}%")
print(f"  Labeled as hate:     {ambig_labeled_hate:,}")
print(f"  Labeled as not-hate: {ambig_labeled_nothate:,}")

# Maximum theoretical F1 assuming these labels are 50/50 correct
# If ~30% of labels have random noise, perfect model gets ~0.70 macro F1
# because it can't predict noise correctly
clean = total - ambiguous
noise_rate = ambiguous / total
print(f"\nClean samples: {clean:,} ({clean/total*100:.1f}%)")
print(f"Noise rate:    {noise_rate*100:.1f}%")
print(f"Theoretical ceiling (if all ambiguous are ~random): ~{(1 - noise_rate/2)*0.87:.3f} macro F1")

# ── 3. Per-split analysis ────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. AGREEMENT LEVEL IN VALIDATION SET")
print("="*60)

val_ids = load_split_ids("val")
val_agree = Counter()
val_ambig_binary = 0
for tid in val_ids:
    if tid in labels:
        val_agree[labels[tid]["agreement_level"]] += 1
        ann = labels[tid]["annotator_labels"]
        if len(set(0 if a == 0 else 1 for a in ann)) > 1:
            val_ambig_binary += 1

for level in [3, 2, 1]:
    n = val_agree[level]
    print(f"  Level {level}: {n:>6,}  ({n/len(val_ids)*100:5.1f}%)")
print(f"  Binary-ambiguous: {val_ambig_binary:,} ({val_ambig_binary/len(val_ids)*100:.1f}%)")

# ── 4. Label distribution by fine-grained category ──────────────────────────
print("\n" + "="*60)
print("4. FINE-GRAINED LABEL DISTRIBUTION (all data)")
print("="*60)

LABEL_NAMES = {0: "NotHate", 1: "Racist", 2: "Sexist", 3: "Homophobe", 4: "Religion", 5: "OtherHate"}
cat_counts = Counter()
for info in labels.values():
    cat_counts[info["hard_label_6class"]] += 1

for c in range(6):
    n = cat_counts[c]
    print(f"  {c} ({LABEL_NAMES[c]:<10}): {n:>7,}  ({n/total*100:5.1f}%)")

# ── 5. Soft label distribution for hate class ────────────────────────────────
print("\n" + "="*60)
print("5. SOFT BINARY LABELS — CONFIDENCE DISTRIBUTION")
print("="*60)

# soft_label_binary = [p_nothate, p_hate]
hate_probs = []
for info in labels.values():
    hate_probs.append(info["soft_label_binary"][1])

# Bucket into ranges
bins = [(0.0, 0.0, "0.00 (pure NotHate)"),
        (0.333, 0.334, "0.33 (1/3 say hate)"),
        (0.667, 0.668, "0.67 (2/3 say hate)"),
        (1.0, 1.01, "1.00 (all say hate)")]

for lo, hi, desc in bins:
    n = sum(1 for p in hate_probs if lo <= p < hi)
    print(f"  P(hate) = {desc}: {n:>7,}  ({n/total*100:5.1f}%)")
print(f"  Total: {total:,}")
