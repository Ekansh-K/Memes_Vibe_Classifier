"""VLM Caption Evaluation — A3

Evaluates Qwen3-VL and MiniCPM-V-2.6 across 3 prompt strategies on a
controlled 200-sample test set to determine the best model+prompt combination
for captioning the ~67K images in MMHS150K that have no OCR text.

Evaluation matrix:
  Models:  Qwen3-VL-4B (FP16) | Qwen3-VL-8B (INT4) | MiniCPM-V-2.6 (INT4)
  Prompts: prompt_1 (generic) | prompt_2 (social media) | prompt_3 (hate-aware)

Output:
  dataset/vlm_eval/
    eval_sample.json            <- 200-image stratified sample metadata
    {model_tag}_{prompt}.json   <- captions for each model x prompt combination
    eval_summary.json           <- comparison table

Usage:
  # Build sample only (no GPU required, run locally):
  python scripts/vlm_caption_eval.py --build-sample-only

  # Full evaluation (run on Kaggle, or use the notebook):
  python scripts/vlm_caption_eval.py [--skip-qwen3-4b] [--skip-qwen3-8b] [--skip-minicpm]

Hardware: Kaggle dual T4 (2x16GB). Models run on GPU 0 sequentially with
          VRAM freed between each model.

API notes for Qwen3-VL (differs from Qwen2.5-VL):
  - Class: Qwen3VLForConditionalGeneration  (not Qwen2_5_VL...)
  - process_vision_info returns 3 values: image_inputs, video_inputs, video_kwargs
  - Requires: transformers>=4.57.0, qwen_vl_utils>=0.0.14
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths (EVAL_DIR created inside main() to avoid import-time side effects) ─
DATASET_DIR     = PROJECT_ROOT / "dataset"
IMG_DIR         = DATASET_DIR / "img_resized"
GT_FILE         = DATASET_DIR / "MMHS150K_GT.json"
OCR_CONSOLIDATED= DATASET_DIR / "ocr_consolidated.json"
PROCESSED_LABELS= DATASET_DIR / "processed_labels.json"
SPLITS_DIR      = DATASET_DIR / "splits"
EVAL_DIR        = DATASET_DIR / "vlm_eval"

SEED         = 42
N_PER_STRATUM= 50   # 4 strata × 50 = 200 total

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS: dict[str, str] = {
    "prompt_1": (
        "Describe this image in one or two detailed sentences. "
        "Focus on the people, objects, text, symbols, and overall scene."
    ),
    "prompt_2": (
        "This image was posted on social media. Describe in detail:\n"
        "1. Any people visible and what they are doing\n"
        "2. Any text, signs, symbols, or logos visible in the image\n"
        "3. Any offensive, hateful, or sensitive imagery (if present)\n"
        "4. The overall tone/mood of the image\n\n"
        "Be factual and objective. Two to three sentences maximum."
    ),
    "prompt_3": (
        "Analyze this social media image for content moderation purposes. "
        "Provide a brief, factual description covering:\n"
        "- Visual content (people, objects, setting)\n"
        "- Any embedded text, memes, or symbols\n"
        "- Whether the image contains potentially harmful stereotypes, "
        "slurs, violent imagery, or hate symbols\n"
        "Answer in 2-3 sentences. Be objective."
    ),
}

# ── Sample building ───────────────────────────────────────────────────────────

def build_eval_sample(eval_dir: Path, n_per_stratum: int = N_PER_STRATUM) -> list[dict]:
    """Build a stratified 200-image evaluation sample from the test split.

    Strata (n_per_stratum each):
      has_ocr_hate    | has_ocr_nothate | no_ocr_hate | no_ocr_nothate
    """
    import random
    random.seed(SEED)

    with open(GT_FILE, encoding="utf-8") as f:
        gt = json.load(f)
    with open(OCR_CONSOLIDATED, encoding="utf-8") as f:
        ocr = json.load(f)
    with open(PROCESSED_LABELS, encoding="utf-8") as f:
        labels = json.load(f)
    with open(SPLITS_DIR / "test_ids.txt", encoding="utf-8") as f:
        test_ids = set(f.read().strip().split("\n"))

    strata_names = {
        (True,  True):  "has_ocr_hate",
        (True,  False): "has_ocr_nothate",
        (False, True):  "no_ocr_hate",
        (False, False): "no_ocr_nothate",
    }
    candidates: dict[tuple[bool, bool], list[str]] = {k: [] for k in strata_names}

    for tid in test_ids:
        if tid not in labels or tid not in gt:
            continue
        if not (IMG_DIR / f"{tid}.jpg").exists():
            continue
        has_ocr = bool(ocr.get(tid, "").strip())
        is_hate = labels[tid]["hard_label_binary"] == 1
        candidates[(has_ocr, is_hate)].append(tid)

    sample: list[dict] = []
    for key, name in strata_names.items():
        pool = candidates[key]
        if not pool:
            print(f"  [WARN] Stratum '{name}' has no candidates!")
            continue
        chosen = random.sample(pool, min(n_per_stratum, len(pool)))
        for tid in chosen:
            sample.append({
                "tweet_id":          tid,
                "stratum":           name,
                "tweet_text":        gt[tid]["tweet_text"],
                "ocr_text":          ocr.get(tid, ""),
                "hard_label_binary": labels[tid]["hard_label_binary"],
                "hard_label_6class": labels[tid]["hard_label_6class"],
                "img_path":          str(IMG_DIR / f"{tid}.jpg"),
            })
        print(f"  Stratum '{name}': {len(chosen)} / {len(pool)} available")

    print(f"Total eval sample: {len(sample)}")
    sample_path = eval_dir / "eval_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Saved eval sample -> {sample_path}")
    return sample

# ── VRAM helpers ──────────────────────────────────────────────────────────────

def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[VRAM] Freed. Allocated: {alloc:.2f} GB")

# ── Result I/O ────────────────────────────────────────────────────────────────

def save_and_summarize(model_tag: str, results: dict[str, dict], eval_dir: Path) -> None:
    for prompt_name, captions in results.items():
        out = eval_dir / f"{model_tag}_{prompt_name}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
        times  = [v["elapsed_s"] for v in captions.values()]
        lengths= [len(v["caption"]) for v in captions.values() if v["caption"]]
        empty  = sum(1 for v in captions.values() if not v["caption"])
        avg_t  = sum(times) / max(len(times), 1)
        avg_l  = sum(lengths) // max(len(lengths), 1)
        print(f"  {model_tag} x {prompt_name}: avg={avg_t:.2f}s | len={avg_l}c | empty={empty} | -> {out.name}")


def generate_eval_summary(eval_dir: Path) -> None:
    summary: dict[str, dict] = {}
    for result_file in sorted(eval_dir.glob("*_prompt_*.json")):
        stem  = result_file.stem
        parts = stem.split("_prompt_")
        if len(parts) != 2:
            print(f"  [WARN] Unexpected filename, skipping: {result_file.name}")
            continue
        model_tag, prompt_num = parts[0], f"prompt_{parts[1]}"
        with open(result_file, encoding="utf-8") as f:
            captions = json.load(f)
        times  = [v["elapsed_s"] for v in captions.values()]
        lengths= [len(v["caption"]) for v in captions.values() if v["caption"]]
        empty  = sum(1 for v in captions.values() if not v["caption"])
        summary[stem] = {
            "model":           model_tag,
            "prompt":          prompt_num,
            "n_samples":       len(captions),
            "avg_time_s":      round(sum(times)   / max(len(times),   1), 3),
            "avg_caption_len": round(sum(lengths)  / max(len(lengths), 1), 1),
            "empty_count":     empty,
            "empty_pct":       round(empty / max(len(captions), 1) * 100, 1),
        }

    out = eval_dir / "eval_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] eval_summary.json -> {out}")
    print(f"\n{'Model+Prompt':<32} {'N':>5} {'AvgTime':>9} {'AvgLen':>8} {'Empty%':>8}")
    print("-" * 66)
    for key, s in sorted(summary.items()):
        print(f"{key:<32} {s['n_samples']:>5} {s['avg_time_s']:>8.2f}s "
              f"{s['avg_caption_len']:>7.0f}c {s['empty_pct']:>7.1f}%")

# ── Shared Qwen3-VL inference loop ───────────────────────────────────────────

def _run_qwen3(
    model_id: str,
    model_tag: str,
    sample: list[dict],
    use_4bit: bool,
) -> dict[str, dict]:
    """Shared inference loop for any Qwen3-VL variant.

    Confirmed working API (from vram_test_corrected.ipynb Kaggle dual-T4 run):
      1. Class  : AutoModelForImageTextToText  (NOT Qwen3VLForConditionalGeneration)
      2. Unpack : image_inputs, video_inputs, video_kwargs = process_vision_info(
                      messages, return_video_kwargs=True)  — 3 values
         Pass **video_kwargs to the processor call.
      3. Device : next(model.parameters()).device  (works for device_map=auto)
      4. FP16 8B: device_map='auto' splits across both GPUs (~17GB total)
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info  # type: ignore

    quant_label = "INT4" if use_4bit else "FP16"
    print(f"\n{'='*60}\nLoading {model_id} ({quant_label}) ...\n{'='*60}")

    # FP16: use device_map='auto' to split 8B across both GPUs
    # INT4: single GPU is sufficient (~6GB)
    load_kwargs: dict = {"device_map": "auto" if not use_4bit else "cuda:0"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
    model.eval()
    # min_pixels / max_pixels control image token budget
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 256, max_pixels=448 * 448
    )
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1e9
        print(f"  GPU{i} VRAM: {alloc:.1f} GB")

    results: dict[str, dict] = {}
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n  {model_tag} x {prompt_name} ...")
        captions: dict[str, dict] = {}

        for item in tqdm(sample, desc=f"{model_tag}/{prompt_name}"):
            img = Image.open(item["img_path"]).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text",  "text":  prompt_text},
            ]}]
            text_in = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # 3-value unpack confirmed working in vram_test_corrected.ipynb
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = processor(
                text=[text_in],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to(next(model.parameters()).device)

            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, max_new_tokens=150)
            elapsed = time.time() - t0

            # Slice generated tokens only (exclude the prompt)
            prompt_len = inputs["input_ids"].shape[1]
            caption = processor.decode(
                out_ids[0][prompt_len:], skip_special_tokens=True
            ).strip()
            captions[item["tweet_id"]] = {"caption": caption, "elapsed_s": round(elapsed, 3)}

        avg_t = sum(v["elapsed_s"] for v in captions.values()) / max(len(captions), 1)
        avg_l = sum(len(v["caption"]) for v in captions.values()) / max(len(captions), 1)
        print(f"  Done. avg_time={avg_t:.2f}s, avg_len={avg_l:.0f}c")
        results[prompt_name] = captions

    del model, processor
    free_vram()
    return results


def run_qwen3_4b(sample: list[dict]) -> dict[str, dict]:
    """Qwen3-VL-4B-Instruct in FP16 (~9GB VRAM)."""
    return _run_qwen3("Qwen/Qwen3-VL-4B-Instruct", "qwen3_4b", sample, use_4bit=False)


def run_qwen3_8b(sample: list[dict]) -> dict[str, dict]:
    """Qwen3-VL-8B-Instruct in 4-bit INT4 (~6GB VRAM)."""
    return _run_qwen3("Qwen/Qwen3-VL-8B-Instruct", "qwen3_8b", sample, use_4bit=True)

# ── MiniCPM-V-2.6 ────────────────────────────────────────────────────────────

def run_minicpm(sample: list[dict]) -> dict[str, dict]:
    """MiniCPM-V-2.6 (INT4) inference for all 3 prompts.

    API notes:
      - Pass the PIL image inside msgs content, NOT as the image= kwarg.
        image= kwarg is the deprecated MiniCPM-V-2.0 API and causes errors in 2.6.
      - model.chat() returns str in 2.6; guard for tuple in edge cases.
    """
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    MODEL_ID = "openbmb/MiniCPM-V-2_6"
    print(f"\n{'='*60}\nLoading {MODEL_ID} (INT4) ...\n{'='*60}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModel.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    results: dict[str, dict] = {}
    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\n  minicpm x {prompt_name} ...")
        captions: dict[str, dict] = {}

        for item in tqdm(sample, desc=f"minicpm/{prompt_name}"):
            img = Image.open(item["img_path"]).convert("RGB")
            msgs = [{"role": "user", "content": [img, prompt_text]}]
            t0 = time.time()
            try:
                result = model.chat(msgs=msgs, tokenizer=tokenizer, max_new_tokens=150)
                caption = result if isinstance(result, str) else result[0]
                caption = caption.strip()
            except Exception as e:
                print(f"\n  [WARN] minicpm failed on {item['tweet_id']}: {e}")
                caption = ""
            elapsed = time.time() - t0
            captions[item["tweet_id"]] = {"caption": caption, "elapsed_s": round(elapsed, 3)}

        avg_t = sum(v["elapsed_s"] for v in captions.values()) / max(len(captions), 1)
        avg_l = sum(len(v["caption"]) for v in captions.values()) / max(len(captions), 1)
        print(f"  Done. avg_time={avg_t:.2f}s, avg_len={avg_l:.0f}c")
        results[prompt_name] = captions

    del model, tokenizer
    free_vram()
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MMHS150K VLM Caption Evaluation (A3)")
    parser.add_argument("--build-sample-only", action="store_true",
                        help="Build the 200-image sample JSON and exit (no GPU needed)")
    parser.add_argument("--skip-qwen3-4b",  action="store_true")
    parser.add_argument("--skip-qwen3-8b",  action="store_true")
    parser.add_argument("--skip-minicpm",   action="store_true")
    args = parser.parse_args()

    # Create eval dir only on actual run (not at import time)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  MMHS150K VLM Caption Evaluation (A3)")
    print("  Models : Qwen3-VL-4B (FP16) | Qwen3-VL-8B (INT4) | MiniCPM-V-2.6 (INT4)")
    print("  Prompts: prompt_1 | prompt_2 | prompt_3")
    print(f"  Sample : {N_PER_STRATUM * 4} images (stratified, test split only)")
    print("=" * 70)

    # ── Load or build eval sample ────────────────────────────────────────────
    sample_path = EVAL_DIR / "eval_sample.json"
    if sample_path.exists():
        print(f"\n[INFO] Using existing eval sample: {sample_path}")
        with open(sample_path, encoding="utf-8") as f:
            sample = json.load(f)
        print(f"       {len(sample)} images")
    else:
        print("\n[INFO] Building stratified eval sample...")
        sample = build_eval_sample(EVAL_DIR)

    if args.build_sample_only:
        print("\n[INFO] --build-sample-only: done.")
        return

    # ── Qwen3-VL-4B ──────────────────────────────────────────────────────────
    tag = "qwen3_4b"
    if args.skip_qwen3_4b:
        print(f"\n[SKIP] {tag} (--skip-qwen3-4b)")
    elif all((EVAL_DIR / f"{tag}_{p}.json").exists() for p in PROMPTS):
        print(f"\n[SKIP] {tag} results already exist")
    else:
        save_and_summarize(tag, run_qwen3_4b(sample), EVAL_DIR)

    # ── Qwen3-VL-8B ──────────────────────────────────────────────────────────
    tag = "qwen3_8b"
    if args.skip_qwen3_8b:
        print(f"\n[SKIP] {tag} (--skip-qwen3-8b)")
    elif all((EVAL_DIR / f"{tag}_{p}.json").exists() for p in PROMPTS):
        print(f"\n[SKIP] {tag} results already exist")
    else:
        save_and_summarize(tag, run_qwen3_8b(sample), EVAL_DIR)

    # ── MiniCPM-V-2.6 ────────────────────────────────────────────────────────
    tag = "minicpm"
    if args.skip_minicpm:
        print(f"\n[SKIP] {tag} (--skip-minicpm)")
    elif all((EVAL_DIR / f"{tag}_{p}.json").exists() for p in PROMPTS):
        print(f"\n[SKIP] {tag} results already exist")
    else:
        save_and_summarize(tag, run_minicpm(sample), EVAL_DIR)

    # ── Summary ───────────────────────────────────────────────────────────────
    generate_eval_summary(EVAL_DIR)
    print(f"\n[DONE] A3 complete. Results in {EVAL_DIR}")


if __name__ == "__main__":
    main()
