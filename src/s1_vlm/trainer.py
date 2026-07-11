"""LoRA fine-tune Qwen3-VL for binary hate Stage-1.

Uses generative classification: model answers 'hateful' or 'not hateful'.
At eval, score = P('hateful') / (P('hateful')+P('not')) from next-token logits when possible;
else parse generated text.
"""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.s1_vlm.config import S1VLMConfig
from src.stage1.dataset import Stage1Dataset, stage1_collate_multimodal
from src.stage1.eval import evaluate_stage1, save_stage1_results

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are a content moderation classifier.\n"
    "Tweet text and image context:\n{text}\n\n"
    "Does this multimodal post contain hate speech "
    "(attacks on race, gender, sexual orientation, religion, or other protected groups)?\n"
    "Answer with exactly one word: hateful or not_hateful."
)


def _device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _load_qwen3vl(model_name: str, use_4bit: bool, device: torch.device):
    """Load Qwen3-VL (or compatible Qwen2-VL) with transformers."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    candidates = [model_name, "Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]
    last = None
    for name in candidates:
        try:
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "device_map": "auto" if device.type == "cuda" else None,
            }
            if use_4bit:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
            if name != model_name:
                logger.info(f"[S1VLM] Loaded fallback model {name}")
            return model, processor, name
        except Exception as e:
            last = e
            logger.warning(f"[S1VLM] Failed to load {name}: {e}")
            continue
    raise RuntimeError(f"Could not load any VLM. Last error: {last}") from last


def _apply_lora(model, config: S1VLMConfig):
    from peft import LoraConfig, get_peft_model, TaskType

    if config.use_4bit:
        try:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model)
        except Exception as e:
            logger.warning(f"[S1VLM] prepare_model_for_kbit_training failed: {e}")

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def _mask_labels_to_answer(input_ids: torch.Tensor, answer: str, tokenizer) -> torch.Tensor:
    """Mask all tokens except the answer suffix for causal LM loss (-100)."""
    labels = input_ids.clone()
    # Default: mask everything; unmask last few tokens that match answer encoding
    labels[:] = -100
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ans_ids:
        # Fallback: train on last 4 tokens
        labels[:, -4:] = input_ids[:, -4:]
        return labels
    seq = input_ids[0].tolist()
    # Find answer subsequence from the end
    n, m = len(seq), len(ans_ids)
    start = None
    for i in range(n - m, -1, -1):
        if seq[i : i + m] == ans_ids:
            start = i
            break
    if start is None:
        labels[:, -max(m, 2) :] = input_ids[:, -max(m, 2) :]
    else:
        labels[0, start : start + m] = input_ids[0, start : start + m]
    return labels


def _label_str(hard: int) -> str:
    return "hateful" if hard == 1 else "not_hateful"


def _parse_pred(text: str) -> float:
    t = text.lower().strip()
    if "not_hateful" in t or "not hateful" in t or re.search(r"\bnot\b", t):
        if "hateful" in t and "not" not in t.split("hateful")[0][-10:]:
            pass
        else:
            return 0.0
    if "hateful" in t or t.startswith("hate"):
        return 1.0
    return 0.5


def run_s1_vlm(config: S1VLMConfig) -> dict:
    _seed(config.seed)
    device = _device(config.device)
    logger.info(f"[S1VLM] Loading {config.model_name} on {device}")

    model, processor, used_name = _load_qwen3vl(
        config.model_name, config.use_4bit, device
    )
    model = _apply_lora(model, config)
    if device.type == "cuda" and not hasattr(model, "hf_device_map"):
        model = model.to(device)

    train_ds = Stage1Dataset(
        "train",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=True,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    val_ds = Stage1Dataset(
        "val",
        text_mode=config.text_mode,
        ocr_source=config.ocr_source,
        load_images=True,
        max_samples=config.max_val_samples,
        seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=stage1_collate_multimodal,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, config.num_workers // 2),
        collate_fn=stage1_collate_multimodal,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.results_run_dir.mkdir(parents=True, exist_ok=True)
    config.save(config.run_dir / "config.yaml")

    model.train()
    global_step = 0
    best_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"VLM ep{epoch}/{config.epochs}")
        for step, batch in enumerate(pbar, 1):
            # Build one sample at a time for VL processor compatibility, accumulate
            batch_loss = 0.0
            n_ok = 0
            for i in range(len(batch["text"])):
                answer = _label_str(int(batch["label_binary"][i].item()))
                user_text = PROMPT_TEMPLATE.format(text=batch["text"][i][:800])
                # Prefer chat template if available
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": batch["image"][i]},
                                {"type": "text", "text": user_text},
                            ],
                        },
                        {"role": "assistant", "content": answer},
                    ]
                    text_prompt = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    inputs = processor(
                        text=[text_prompt],
                        images=[batch["image"][i]],
                        return_tensors="pt",
                        padding=True,
                    )
                except Exception:
                    # Minimal fallback: text-only if multimodal packaging fails
                    inputs = processor(
                        text=[user_text + "\nAnswer: " + answer],
                        images=[batch["image"][i]],
                        return_tensors="pt",
                        padding=True,
                    )

                inputs = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in inputs.items()
                }
                tok = getattr(processor, "tokenizer", processor)
                labels = _mask_labels_to_answer(
                    inputs["input_ids"], answer, tok
                )
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss / (config.grad_accum_steps * max(len(batch["text"]), 1))
                loss.backward()
                batch_loss += float(loss.item())
                n_ok += 1

            if step % config.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += batch_loss
            pbar.set_postfix(loss=f"{running / step:.4f}")

        # Lightweight val every epoch (generate)
        model.eval()
        y_true, y_prob, ids = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="VLM val", leave=False):
                for i in range(len(batch["text"])):
                    user_text = PROMPT_TEMPLATE.format(text=batch["text"][i][:800])
                    try:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": batch["image"][i]},
                                    {"type": "text", "text": user_text},
                                ],
                            }
                        ]
                        text_prompt = processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        inputs = processor(
                            text=[text_prompt],
                            images=[batch["image"][i]],
                            return_tensors="pt",
                            padding=True,
                        )
                    except Exception:
                        inputs = processor(
                            text=[user_text + "\nAnswer:"],
                            images=[batch["image"][i]],
                            return_tensors="pt",
                            padding=True,
                        )
                    inputs = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in inputs.items()
                    }
                    gen = model.generate(**inputs, max_new_tokens=8, do_sample=False)
                    out_ids = gen[0][inputs["input_ids"].shape[1] :]
                    text_out = processor.decode(out_ids, skip_special_tokens=True)
                    y_prob.append(_parse_pred(text_out))
                    y_true.append(int(batch["label_binary"][i].item()))
                    ids.append(batch["tweet_id"][i])
        y_true_a = np.array(y_true)
        y_prob_a = np.array(y_prob, dtype=float)
        metrics = evaluate_stage1(y_true_a, y_prob_a)
        metrics["epoch"] = epoch
        metrics["model_name"] = used_name
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_stage1_results(config.results_run_dir, metrics, y_true_a, y_prob_a, ids)
            model.save_pretrained(config.run_dir / "lora_adapter")
            processor.save_pretrained(config.run_dir / "processor")
            logger.info(f"[S1VLM] New best macro_f1={best_f1:.4f} — saved adapter")
        model.train()
        logger.info(
            f"[S1VLM] epoch {epoch}: macro_f1={metrics['macro_f1']:.4f} "
            f"hate_recall={metrics['hate_recall']:.4f}"
        )

    metrics["run_name"] = config.run_name
    metrics["best_macro_f1"] = best_f1
    return metrics
