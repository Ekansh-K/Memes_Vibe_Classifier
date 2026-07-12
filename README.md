# Memes Vibe Classifier

Hello! This is my project — an attempt to tackle **[MMHS150K](https://gombru.github.io/2019/10/09/MMHS/)**, one of the largest multimodal hate-speech datasets (~150k tweet + image pairs).

The goal is simple to say and hard to get right: look at a meme (image + text), decide whether it is hateful, and if it is, figure out **what kind** of hate it is.

Most papers on this dataset stop at a single binary head. I built a full **two-stage pipeline** so Stage 2 is not an afterthought — once something is flagged as hate, a second head predicts the fine-grained type(s).

---

## The problem

Memes are multimodal. Tweet text can look harmless alone, the image can look harmless alone, and together they form a racist joke or a sexist punchline. The model has to use **both**.

| Challenge | What shows up in MMHS150K |
|---|---|
| Class imbalance | ~78–83% NotHate; Religion is ~0.1–0.2% of labels |
| Annotator noise | Only ~41–44% of samples have unanimous hate / not-hate agreement |
| Multi-label types | A meme can be Racist **and** Sexist at the same time |
| Accuracy alone is misleading | Always predicting NotHate looks “good” and fails the real task |

I report **F1** (not accuracy alone).

![MMHS150K label distributions and annotator agreement](assets/label_distributions.png)

---

## How Stage 1 compares to other work on MMHS150K

Stage 1 is the binary gate: **Hate vs NotHate**. That is the setup almost every paper reports on this dataset, so it is the fairest place to compare.

| Method | Year | Setup (brief) | F1 |
|---|---|---|---|
| Random baseline ([Gomez et al., WACV 2020](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf)) | 2020 | Chance-level | 0.67 |
| Image-only FCM ([Gomez et al.](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf)) | 2020 | Inception-v3 | 0.67 |
| LSTM text ([Gomez et al.](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf)) | 2020 | GloVe + LSTM on tweet text | 0.70 |
| Davison-style text ([Gomez et al.](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf) retrain) | 2020 | Text baseline from earlier HS work | 0.70 |
| FCM multimodal ([Gomez et al.](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf)) | 2020 | Inception-v3 + LSTM (tweet + OCR) | **0.70** |
| SCM / TKM multimodal ([Gomez et al.](https://openaccess.thecvf.com/content_WACV_2020/papers/Gomez_Exploring_Hate_Speech_Detection_in_Multimodal_Publications_WACV_2020_paper.pdf)) | 2020 | Spatial / textual-kernel fusion | 0.70 |
| Text / simple multimodal re-runs (typical later papers) | 2021–24 | BERT / CNN fusion variants | ~0.55–0.68 |
| Ensemble InceptionV3 + BERT + XLNet ([Kashif et al., CASE 2023](https://aclanthology.org/2023.case-1.12.pdf)) | 2023 | Fused image + text ensemble | **0.75** |
| Stronger CLIP / contrastive / prompting methods (literature ballpark) | 2022–25 | Hate-CLIPper-style / RGCL-style ideas, often on related meme sets | ~0.70+ |
| **This project — Stage 1 best** | 2026 | Text + Hate-CLIPper-style stack, ensemble | **0.71** |
| **This project — P2-TCAM Stage 1** | 2026 | CLIP ViT-L/14 + TweetEval RoBERTa + TCAM | **0.71** |

**Takeaways from the comparison**

- Gomez et al. (the original MMHS150K paper) already showed that **text is strong** and early multimodal fusion barely beats text-only (~0.70 F1). That is still the honest baseline on this set.
- Image-only is weak (~0.67 F1) — hate is rarely in the pixels alone.
- Later ensemble work (e.g. Kashif et al.) can push into the mid-0.70s by stacking strong unimodal models.
- Published binary scores on MMHS150K mostly sit around **0.55–0.75 F1**. Scores above that often use different splits, different label collapses, or accuracy instead of F1.
- My Stage 1 lands at **0.71 F1** — in line with Gomez’s best multimodal F1 and competitive with the common literature band. I am not claiming a new SOTA; I am saying Stage 1 is in the right range for this noisy dataset.

Why Stage 1 is hard to push further: more than half the binary labels are majority-vote (2/3) ambiguous. Past a point you are fitting annotator noise, not a clean “true” hate signal.

---

## What I built

### Two-stage pipeline

```
meme image + (tweet | OCR | caption)
              │
              ▼
     ┌────────────────────┐
     │  Multimodal encoder │
     └─────────┬──────────┘
               │
       Stage 1 │  Hate vs NotHate
               │
         if Hate
               │
       Stage 2 │  Racist · Sexist · Homophobe · Religion · OtherHate
                 (multi-label — more than one type can fire)
```

- **Stage 1** decides if the meme is hateful at all.
- **Stage 2** only runs on the hate path and predicts type(s).
- Stage 2 can reject a bad Stage-1 false positive (all types off). It cannot recover a Stage-1 miss.

### Architecture — P2-TCAM

**P2-TCAM** = Text-guided Cross-Attention Multimodal:

| Branch | Backbone | Role |
|---|---|---|
| Vision | Frozen **CLIP ViT-L/14** | Image patches → 768-d |
| Text | **TweetEval RoBERTa** (last layers unfrozen) | Tweet + OCR (+ optional caption) |
| Fusion | **TCAM** cross-attention | Visual queries attend to text keys/values |
| Heads | Early fusion → Stage 1 + Stage 2 | Binary, then 5-type multi-label |

![P2-TCAM model flow](assets/p2_architecture.png)

Training choices that mattered on this set:

- Class weighting for the NotHate-heavy split  
- Agreement-aware loss when annotators disagree  
- Threshold sweep on val (0.5 is a bad default here)  
- Per-type Stage-2 thresholds (e.g. Racist ~0.50, rarer types higher)

I also trained a dedicated **Stage-1 stack** (full fine-tune text models, Hate-CLIPper-style align fusion, light VLM LoRA, probability ensemble) to push the binary gate without rewriting Stage 2.

---

## Results

### Full pipeline (P2-TCAM, Variation D, `all_text`)

| Stage | Task | F1 |
|---|---|---|
| Stage 1 | Hate vs NotHate | **0.71** |
| Stage 2 | 5-type multi-label | **0.87** |

Stage 2 per-type F1:

| Racist | Sexist | Homophobe | Religion | OtherHate |
|---|---|---|---|---|
| 0.92 | 0.74 | 0.91 | 0.67 | 0.80 |

Religion stays the hardest class — almost no training mass. Racist / Homophobe are where multimodal signal helps most.

![Stage F1 and Stage-2 per-type F1](assets/pipeline_summary.png)

Stage 2 is the stronger half once hate is detected. Most papers never report this second stage cleanly on MMHS150K; that is the main design choice of this project.

### Stage-1 ablations (binary only)

| Model | F1 |
|---|---|
| Text — TweetEval RoBERTa | 0.70 |
| Text — hate-specialized FT | 0.71 |
| Soft / hard label recipes | up to ~0.71 |
| Hate-CLIPper-style (align + adapters) | ~0.70 |
| Ensemble of full-val members | **0.71** |

Text alone is already strong (same lesson as Gomez). Multimodal fusion and ensembling help a little; VLM LoRA helped less than text + CLIP-style fusion on this set.

---

## Future work

If I push this further, the next lever is less “train longer on the same loss” and more better representation learning under noise:

- **RGCL-style** retrieval / contrastive losses (as in RGCL-HateCLIPper-type work) — pull same-class multimodal pairs together and hard negatives apart  
- Cleaner OCR / caption filtering  
- Treating Religion almost as few-shot instead of a full head  

---

## Dataset & references

- **Dataset:** [MMHS150K](https://gombru.github.io/2019/10/09/MMHS/) — Gomez et al., *Exploring Hate Speech Detection in Multimodal Publications* (WACV 2020)  
- Gomez et al. baselines (LSTM / FCM / SCM / TKM) — same paper, Table 1  
- Kashif et al., *Multimodal Hate Speech Detection using Fused Ensemble Learning* (CASE 2023)  
- Related ideas: Hate-CLIPper (cross-modal CLIP fusion), MemeCLIP-style adapters, RGCL-style contrastive multimodal hate detection  

---

Built as my Deep Learning end-sem project on a noisy, imbalanced, real multimodal hate-speech benchmark — with Stage 2 treated as first-class, not a footnote.
