# Data Preprocessing & Enrichment Details

This document explains the data preprocessing, Optical Character Recognition (OCR) improvements, and Visual-Language Model (VLM) caption generation pipeline developed for the MMHS-150K dataset.

---

## 1. OCR Extraction & Improvements

### The Problem with Stale OCR
The original MMHS-150K dataset shipped with OCR text (located in `dataset/img_txt/`) that suffered from three major issues:
1. **Low Coverage**: Only **59,252 out of 150,000 images** had OCR text (~39.5%). Over 90,000 images had zero text extracted.
2. **Garbled Quality**: The old OCR (processed around 2018) produced garbled and noisy text due to lack of support for stylized/angled fonts. For example:
   * *Old OCR:* `İ'M SLOWLY BEC«MİNG RETARpEp!`
   * *Actual Text:* `I'M SLOWLY BECOMING RETARDED`
3. **Outdated Engines**: Early OCR engines struggled with the complex background noise, overlapping color contrast, and meme templates common in internet graphics.

### The New EasyOCR Pipeline
To fix this, we developed an extraction pipeline (`scripts/preprocess_ocr.py`) using **EasyOCR** powered by a local GPU (NVIDIA GTX 1650):

* **Model**: EasyOCR (built on PyTorch, using a CRAFT detector and a ResNet/LSTM recognizer).
* **Execution**: Run on the local GPU, outputting a separate `.json` file for every processed image containing detected text blocks, bounding boxes, and confidence levels.
* **Consolidation**: Once the image-by-image extraction completes, the JSON results are compiled into a unified map file: `dataset/ocr_consolidated.json`.

### Quantitative Results & Impact
* **OCR Coverage Increase**: Non-empty OCR coverage increased from **39.5%** to **54.9%** (representing **80,683 / 150,000 images**).
* **Text Resolution**: Memes that previously had zero text now have clean, readable text. The overall OCR string quality is dramatically cleaner, removing non-ASCII artifacts and character misclassifications.
* **Preprocessing Rules**: In the trainer, the OCR text undergoes clean-up (`clean_ocr_text`):
  * Removal of trailing/leading brackets or non-alphanumeric noise.
  * Stripping of extra whitespaces.
  * Integration into text loaders using the `[SEP]` boundary token.

---

## 2. VLM Caption Generation

To enrich the model's visual reasoning capabilities, we generate text descriptions (captions) of the meme images. These captions act as a bridge, allowing the text encoder (RoBERTa) to parse high-level visual concepts that are hard to capture via raw patch tokens.

### A. VLM Model Evaluation & Prompt Matrix
We conducted a controlled evaluation (`scripts/vlm_caption_eval.py`) on a stratified sample of 200 test set images to find the best model and prompting strategy.

#### Models Evaluated
1. **Qwen3-VL-4B-Instruct (FP16)** — Loaded using `AutoModelForImageTextToText` (~9GB VRAM).
2. **Qwen3-VL-8B-Instruct (INT4)** — Quantized version using BitsAndBytes (~6GB VRAM).
3. **MiniCPM-V-2.6 (INT4)** — Quantized using BitsAndBytes (~6GB VRAM).

#### Prompt Strategies Swept
* **Prompt 1 (Generic Description)**:
  > *"Describe this image in one or two detailed sentences. Focus on the people, objects, text, symbols, and overall scene."*
* **Prompt 2 (Social Media Focus)**:
  > *"This image was posted on social media. Describe in detail: 1. Any people visible and what they are doing. 2. Any text, signs, symbols, or logos visible in the image. 3. Any offensive, hateful, or sensitive imagery (if present). 4. The overall tone/mood of the image. Be factual and objective. Two to three sentences maximum."*
* **Prompt 3 (Moderation/Hate-Aware Focus)**:
  > *"Analyze this social media image for content moderation purposes. Provide a brief, factual description covering: Visual content (people, objects, setting), Any embedded text, memes, or symbols, Whether the image contains potentially harmful stereotypes, slurs, violent imagery, or hate symbols. Answer in 2-3 sentences. Be objective."*

### B. VLM Evaluation Results
The models were benchmarked on a dual-T4 Kaggle GPU setup for inference speed, caption length, and failure/empty rates:

| Model + Prompt | Avg Inference Time | Avg Caption Length | Empty Rate % |
| :--- | :---: | :---: | :---: |
| **Qwen3-VL-4B x Prompt 3** | ~1.85s | ~210 chars | 0.0% |
| **Qwen3-VL-8B x Prompt 3** | ~2.40s | ~240 chars | 0.0% |
| **MiniCPM-V-2.6 x Prompt 2**| ~3.10s | ~180 chars | 0.2% |

**Selection**: The final captions compiled into `results/vlm_captions.json` were generated using **Qwen3-VL-8B-Instruct** combined with the Content Moderation prompt (**Prompt 3**). This combination provided the most objective, hate-aware descriptions without hallucinating or outputting empty strings.

---

## 3. Data Flow & VLM Data Integration

Once the consolidated OCR and VLM captions are generated, they are integrated into the downstream training data files (`dataset/vlm_train_binary.jsonl`, etc.) using the LLaVA conversation format:

```json
{
  "id": "1103982848201",
  "image": "dataset/img_resized/1103982848201.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nTweet: \"Look at these clowns coming over the border\"\nText visible in image: \"IMMIGRATION DEPT\"\n\nAnalyze this tweet and associated image for hate speech. Consider both textual content and visual context. Is this hateful or not hateful?"
    },
    {
      "from": "gpt",
      "value": "hateful"
    }
  ]
}
```

### Downstream Fusion Logic
When running the multimodal TCAM pipeline under `text_mode="all_text"`, the dataset loader constructs the text input representation as:
$$\text{Fused Text} = \text{VLM\_Caption} + \text{ " [SEP] " } + \text{OCR\_Text} + \text{ " [SEP] " } + \text{Tweet\_Text}$$

This combines three layers of semantic meaning:
1. **The VLM Caption**: Provides structural scene understanding (e.g., *"An illustration of two men wearing hats..."*).
2. **The OCR Text**: Captures exact textual punches written inside the graphic.
3. **The Tweet Text**: Captures the social media container's original context, hashtags, and emojis.
