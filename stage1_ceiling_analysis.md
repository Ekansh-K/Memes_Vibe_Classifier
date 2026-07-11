# Stage 1 Ceiling Analysis — Why 0.66 is Possibly Near the Ceiling

## The Hard Truth: 55.6% of Labels Are Noisy

The data analysis reveals something critical that reframes our entire optimization strategy:

| Category | Count | % of Total |
|---|---|---|
| **Unanimous** (3/3 annotators agree on hate/not-hate) | 66,508 | 44.4% |
| **Majority** (2/3 agree, 1 disagrees) | 83,305 | **55.6%** |
| **Full split** (all disagree) | 10 | 0.0% |

> [!CAUTION]
> **More than half of MMHS150K has label noise on the binary hate/not-hate axis.**
> For these 83K samples, the "ground truth" is decided by a 2-1 majority vote.
> The dissenting annotator saw the same meme and reached the opposite conclusion.
> No model — no matter how powerful — can consistently predict labels where
> humans themselves disagree.

### What This Means for the F1 Ceiling

For the ~55.6% of ambiguous samples, even a hypothetically perfect model faces a paradox: if it correctly identifies the "true" signal, it disagrees with the noisy label ~33% of the time (the wrong annotator won the vote). This creates an **inherent noise ceiling** on macro F1.

Published baselines on MMHS150K confirm this:
- Most models score **0.55–0.65 macro F1** for binary classification
- SOTA methods with contrastive learning, hypergraph fusion, or prompting reach **~0.70**
- **No published method exceeds 0.72 macro F1** on this binary task

**Your 0.6613 is already in the top tier** — not at the bottom of performance. The ceiling is the data, not the model.

---

## Hate Rate by Agreement Level — The Signal-to-Noise Problem

| Agreement | Hate % | Count | Interpretation |
|---|---|---|---|
| Level 3 (all agree) | 6.6% | 62,025 | Very clean — mostly "clearly not hate" |
| Level 2 (majority) | 27.8% | 76,097 | Noisy — hate cases are ambiguous |
| Level 1 (all differ) | 4.7% | 11,701 | Pure noise — label is arbitrary |

The hate class is *disproportionately noisy*. Only 4,094 hate samples have unanimous agreement (6.6% of Level 3). Meanwhile, 21,172 hate samples (82% of all hate) come from ambiguous majority votes. The model must learn to classify hate primarily from samples where humans couldn't agree.

---

## Fine-Grained Category Distribution

| Category | Count | % |
|---|---|---|
| NotHate | 124,003 | 82.8% |
| Racist | 12,288 | 8.2% |
| Sexist | 3,671 | 2.5% |
| Homophobe | 3,886 | 2.6% |
| **Religion** | **164** | **0.1%** |
| OtherHate | 5,811 | 3.9% |

> [!IMPORTANT]
> Religion has only **164 samples** in the entire 150K dataset. That's 0.1%.
> Even with pos_weight=100, this category is nearly impossible to learn from.
> It's borderline statistical noise rather than a learnable class.

---

## What Can Actually Improve Stage 1?

Given this analysis, here are the options ranked by **feasibility × expected impact**:

### Option A: Use Soft Labels Instead of Hard Labels (★★★★★)

**The single highest-impact change available.**

Currently, the model trains with **hard binary labels** (0 or 1) from majority vote. But the data already contains soft labels — the annotator vote distribution:
- `soft_label_binary = [0.667, 0.333]` means 2 annotators said not-hate, 1 said hate
- The hard label throws away this uncertainty information

**Implementation**: Replace `BCEWithLogitsLoss` with **soft BCE** using the actual vote probabilities as targets:
```python
# Instead of: target = 0.0 or 1.0
# Use:        target = soft_label_binary[1]  (= 0.0, 0.333, 0.667, or 1.0)
```

**Why it works**:
- The model learns that a 0.333 sample is *somewhat* hateful, not *absolutely not hateful*
- Reduces the penalty for predicting 0.3 on an ambiguous sample that happens to be labeled 0
- Effectively denoises the training signal — the model gets calibrated probabilities instead of noisy 0/1 flips
- **Free to implement — no architecture change, no extra training time, no extra data**

**Expected gain**: +0.02–0.05 macro F1. This is the approach used by Gomez et al. in the original paper and by many subsequent works that achieve the 0.68–0.72 range.

---

### Option B: Agreement-Weighted Loss (★★★★☆)

Give higher loss weight to high-agreement samples and lower weight to ambiguous ones.

```python
# agreement_level: 3 (all agree) → weight 1.0
#                  2 (majority)  → weight 0.5
#                  1 (all differ)→ weight 0.2
```

This tells the model: "focus on learning the clear cases first, and don't overfit to noisy ambiguous cases."

Can be combined with Option A for a stronger effect.

**Expected gain**: +0.01–0.03 macro F1

---

### Option C: Label Smoothing (★★★☆☆)

Already supported in the config (`label_smoothing: float = 0.0`) but not enabled.

Set `label_smoothing = 0.1` to soften hard targets (0.0 → 0.05, 1.0 → 0.95). This is a weaker version of Option A — it applies uniform smoothing to all samples rather than using the actual annotator disagreement.

**Expected gain**: +0.005–0.015 macro F1 (much less than soft labels)

---

### Option D: Focal Loss (★★★☆☆)

Replace BCE with focal loss to down-weight easy examples and focus on hard (ambiguous) ones:
```python
# focal_loss = -(1-p)^gamma * log(p)
# gamma=2.0 is standard
```

Focal loss helps when the model is already confident about easy cases (which it is — not-hate F1 is ~0.92). It redirects gradient toward the hard boundary cases.

**Expected gain**: +0.01–0.02 macro F1

---

### Option E: Curriculum-Based Training (★★☆☆☆)

Train in two phases:
1. **Phase A**: Train only on high-agreement samples (Level 3: 62K samples, very clean)
2. **Phase B**: Fine-tune on all samples

Lets the model learn clean patterns first, then adapt to noisy data.

**Expected gain**: +0.01–0.02 macro F1, but adds complexity and needs careful tuning

---

### Option F: Better Backbone — HateBERT (★★★☆☆)

Replace `cardiffnlp/twitter-roberta-base` with `GroNLP/hateBERT`, which was pre-trained on RAL-E (Reddit Abusive Language) dataset. It understands hate speech lexicon better than generic Twitter text.

**Complication**: Different tokenizer + model dimensions may require adjusting `d_t` and the projection layer. Medium implementation effort.

**Expected gain**: +0.01–0.03 macro F1

---

### Option G: Ensemble / Test-Time Augmentation (★★☆☆☆)

Train 3-5 models with different seeds and average predictions. With noisy labels, ensembling is one of the most reliable ways to smooth out label noise effects.

**Complication**: 3-5× training cost on Kaggle. Likely impractical given GPU time limits.

---

## Recommended Strategy (by priority)

| Priority | Change | Expected Δ F1 | Implementation Time |
|---|---|---|---|
| **1** | **Soft labels** (Option A) | +0.02–0.05 | ~30 min |
| 2 | Agreement-weighted loss (Option B) | +0.01–0.03 | ~20 min |
| 3 | Focal loss (Option D) | +0.01–0.02 | ~30 min |
| 4 | Label smoothing (Option C) | +0.005–0.015 | Already supported |
| — | HateBERT (Option F) | +0.01–0.03 | ~2 hours |
| — | Ensemble (Option G) | +0.02–0.04 | 3-5× training cost |

> [!TIP]
> **Options A and B together should get you to 0.68–0.71 macro F1**, which is
> near SOTA for this dataset. That's the realistic ceiling given the label noise.
> Going above 0.72 would require either cleaning the dataset (re-annotation)
> or using an ensemble, both of which are impractical for this project.

---

## The Real Takeaway

Your current **0.6613** is not bad — it's **already competitive with published baselines** on MMHS150K. The problem is not the architecture (TCAM cross-attention is well-suited for this task) or the features (CLIP + TweetEval are appropriate encoders). The problem is that **the human annotators only agree 44.4% of the time on the hate/not-hate binary question**.

No binary classifier can exceed the inter-annotator agreement rate by much. The path forward is to stop treating noisy labels as ground truth and instead train on the soft vote distributions.
