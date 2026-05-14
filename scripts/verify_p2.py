"""P2 verification script — run before full training to catch issues early.

Checks:
1. All imports resolve correctly
2. P2Config serialises/deserialises correctly
3. P2Dataset loads and returns correct sample structure
4. TCAM forward pass produces correct output shapes
5. CLIP patch tokens are (B, 257, 768)
6. CLIP and TweetEval are frozen (no requires_grad)
7. proj_t weight is identity-initialised
8. Stage 2 reinit: proj_t unchanged, cross_attn/head reset
9. Loss factory returns correct type per variation
10. Collate function handles mixed PIL/str batch

Usage:
    python scripts/verify_p2.py
    python scripts/verify_p2.py --max_samples 50   # faster
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)  # suppress noisy HF/CLIP logs during verify

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
results = []


def check(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL} {name}")
        print(f"         {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()


def skip(name: str, reason: str):
    results.append((SKIP, name))
    print(f"  {SKIP} {name} ({reason})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nP2 Verification  |  device={device}\n{'='*50}")

    # ── 1. Imports ────────────────────────────────────────────────────────────
    print("\n[1] Import checks")

    def check_imports():
        from src.p2.config import P2Config
        from src.p2.dataset import P2Dataset, p2_collate_fn, HATE_CAT_NAMES
        from src.p2.model import TCAM, CrossAttention
        from src.p2.losses import get_p2_loss
        from src.p2.metrics import (
            compute_binary_metrics, compute_multiclass_metrics,
            compute_multilabel_metrics, calibrate_thresholds,
            apply_thresholds, compute_pipeline_metrics,
        )
        from src.p2.trainer import run_p2, train_stage

    check("All P2 modules import", check_imports)

    # ── 2. Config ─────────────────────────────────────────────────────────────
    print("\n[2] Config checks")

    def check_config_defaults():
        from src.p2.config import P2Config
        cfg = P2Config()
        assert cfg.variation == "D"
        assert cfg.text_mode == "all_text"
        assert cfg.s1_batch_size == 16
        assert cfg.grad_accum_steps == 8
        assert cfg.s1_epochs == 5
        assert cfg.s2_epochs == 7

    check("Config defaults correct", check_config_defaults)

    def check_config_autoname():
        from src.p2.config import P2Config
        cfg = P2Config(variation="C", text_mode="tweet_ocr")
        assert cfg.run_name == "p2_C_tweet_ocr", f"Got: {cfg.run_name}"

    check("Config auto-generates run_name", check_config_autoname)

    def check_config_properties():
        from src.p2.config import P2Config
        cfg = P2Config(variation="D")
        assert cfg.num_classes_s1 == 1
        assert cfg.num_classes_s2 == 5
        assert cfg.num_classes_direct == 6
        assert cfg.is_multilabel is True
        assert cfg.is_two_stage is True

    check("Config derived properties", check_config_properties)

    def check_config_yaml():
        import tempfile, os
        from src.p2.config import P2Config
        cfg = P2Config(variation="B", text_mode="tweet_ocr")
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = Path(f.name)
        try:
            cfg.save(path)
            loaded = P2Config.load(path)
            assert loaded.variation == "B"
            assert loaded.text_mode == "tweet_ocr"
        finally:
            path.unlink(missing_ok=True)

    check("Config YAML save/load roundtrip", check_config_yaml)

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    print("\n[3] Dataset checks")

    from src.p2.config import P2Config
    cfg_ds = P2Config(variation="D", text_mode="all_text")
    cfg_ds.max_train_samples = args.max_samples
    cfg_ds.max_val_samples = 20

    def check_dataset_val_loads():
        from src.p2.dataset import P2Dataset
        ds = P2Dataset("val", cfg_ds, stage=None, max_samples=20)
        assert len(ds) > 0, "Empty dataset"
        sample = ds[0]
        from PIL import Image
        assert isinstance(sample["image"], Image.Image), "image should be PIL"
        assert isinstance(sample["text"], str), "text should be str"
        assert isinstance(sample["label_binary"], int)
        assert sample["label_binary"] in (0, 1)
        assert sample["multi_label_binary"].shape == (6,)

    check("P2Dataset loads val split", check_dataset_val_loads)

    def check_dataset_text_modes():
        from src.p2.dataset import P2Dataset
        cfg_tw = P2Config(variation="D", text_mode="tweet_ocr")
        cfg_at = P2Config(variation="D", text_mode="all_text")
        ds_tw = P2Dataset("val", cfg_tw, stage=None, max_samples=50)
        ds_at = P2Dataset("val", cfg_at, stage=None, max_samples=50)

        # Check at least one sample has [SEP] when tweet+ocr are both non-empty
        found_sep_tw = False
        found_sep_at = False
        for i in range(min(50, len(ds_tw))):
            tweet_id = ds_tw.sample_ids[i]
            entry = ds_tw.gt_data[tweet_id]
            ocr = ds_tw.ocr_data.get(tweet_id, "")
            if entry.get("tweet_text", "").strip() and ocr.strip():
                s = ds_tw[i]["text"]
                if "[SEP]" in s:
                    found_sep_tw = True
                    break

        for i in range(min(50, len(ds_at))):
            tweet_id = ds_at.sample_ids[i]
            caption = ds_at.captions.get(tweet_id, "")
            ocr = ds_at.ocr_data.get(tweet_id, "")
            if caption.strip() or ocr.strip():
                s = ds_at[i]["text"]
                if "[SEP]" in s:
                    found_sep_at = True
                    break

        assert found_sep_tw, "tweet_ocr: no sample with [SEP] found in 50 samples"
        assert found_sep_at, "all_text: no sample with [SEP] found in 50 samples"
        print(f"         tweet_ocr sample: {ds_tw[0]['text'][:80]!r}")
        print(f"         all_text  sample: {ds_at[0]['text'][:80]!r}")

    check("Dataset text_mode construction", check_dataset_text_modes)

    def check_dataset_stage2_filter():
        from src.p2.dataset import P2Dataset
        ds_s2 = P2Dataset("val", cfg_ds, stage=2, max_samples=50)
        # All samples must have hard_label_binary == 1
        for sid in ds_s2.sample_ids:
            label = ds_s2.labels[sid]["hard_label_binary"]
            assert label == 1, f"Stage 2 contains non-hateful sample: {sid}"

    check("Stage 2 dataset contains only hateful samples", check_dataset_stage2_filter)

    def check_collate():
        from src.p2.dataset import P2Dataset, p2_collate_fn
        ds = P2Dataset("val", cfg_ds, stage=None, max_samples=4)
        samples = [ds[i] for i in range(min(4, len(ds)))]
        batch = p2_collate_fn(samples)
        assert isinstance(batch["image"], list)
        assert isinstance(batch["text"], list)
        assert batch["label_binary"].shape == (len(samples),)
        assert batch["multi_label_binary"].shape == (len(samples), 6)

    check("Collate function returns correct types/shapes", check_collate)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    print("\n[4] Model checks (may take ~30s to load CLIP + TweetEval)")

    from src.p2.config import P2Config
    cfg_m = P2Config(variation="D")

    # Try to load CLIP — if not installed (local env), skip all model checks gracefully
    try:
        import clip as _clip_test  # noqa
        clip_available = True
    except ImportError:
        clip_available = False

    _CLIP_SKIP_REASON = "CLIP not installed locally (install on Kaggle: pip install git+https://github.com/openai/CLIP.git)"

    if not clip_available:
        model_loaded = False
        model = None
        for _chk in [
            "TCAM loads (num_classes=1)",
            "Parameter counts (trainable << frozen)",
            "CLIP backbone is frozen",
            "TweetEval backbone is frozen",
            "proj_t is identity-initialised",
            "CLIP patch tokens shape (B, 257, 768)",
            "Forward pass shape: binary (B, 1)",
            "Forward pass shape: 6-class (B, 6)",
            "Stage 2 reinit: proj_t kept, cross_attn/head reset",
        ]:
            skip(_chk, _CLIP_SKIP_REASON)
    else:
        # Load once for all model tests
        try:
            from src.p2.model import TCAM
            model = TCAM.from_config(cfg_m, num_classes=1).to(device)
            model_loaded = True
            results.append((PASS, "TCAM loads (num_classes=1)"))
            print(f"  {PASS} TCAM loads (num_classes=1)")
        except Exception as e:
            model_loaded = False
            model = None
            results.append((FAIL, "TCAM loads (num_classes=1)"))
            print(f"  {FAIL} TCAM loads (num_classes=1)")
            print(f"         {type(e).__name__}: {e}")

    if model_loaded:
        def check_param_counts():
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in model.parameters())
            n_frozen = n_total - n_train
            print(f"         Trainable: {n_train:,}  Frozen: {n_frozen:,}  Total: {n_total:,}")
            assert n_train > 0, "No trainable parameters"
            assert n_frozen > n_train, "More trainable than frozen params — backbones not frozen"

        check("Parameter counts (trainable << frozen)", check_param_counts)

        def check_clip_frozen():
            for p in model._clip_model.parameters():
                assert not p.requires_grad, "CLIP parameter has requires_grad=True"

        check("CLIP backbone is frozen", check_clip_frozen)

        def check_tweet_frozen():
            for p in model.tweet_encoder.parameters():
                assert not p.requires_grad, "TweetEval parameter has requires_grad=True"

        check("TweetEval backbone is frozen", check_tweet_frozen)

        def check_proj_t_identity():
            import torch.nn.functional as F
            w = model.proj_t.weight.data.cpu()   # (d_v=1024, d_t=768)
            d_t = w.shape[1]                      # 768
            # Top-left d_t×d_t block should be identity
            top_block = w[:d_t, :d_t]
            eye = torch.eye(d_t)
            mse_top = F.mse_loss(top_block, eye).item()
            # Bottom (d_v-d_t)×d_t block should be zeros
            bottom_block = w[d_t:, :]
            mse_bot = bottom_block.abs().max().item()
            bias_norm = model.proj_t.bias.data.norm().item()
            print(f"         proj_t shape: {list(w.shape)}  (expect [1024, 768])")
            print(f"         top-left {d_t}×{d_t} block MSE from identity: {mse_top:.2e}")
            print(f"         bottom block max abs (expect ~0): {mse_bot:.2e}")
            print(f"         bias norm (expect ~0): {bias_norm:.2e}")
            assert mse_top < 1e-6, f"proj_t top block not identity-init: MSE={mse_top}"
            assert mse_bot < 1e-6, f"proj_t bottom block not zeros: max={mse_bot}"
            assert bias_norm < 1e-6, f"proj_t bias not zero: norm={bias_norm}"

        check("proj_t is partial-identity-initialised (top block=I, bottom=0)", check_proj_t_identity)

        def check_patch_tokens_shape():
            from PIL import Image as PILImage
            imgs = [PILImage.new("RGB", (224, 224), (100, 150, 200)) for _ in range(2)]
            imgs_t = model._preprocess_images(imgs, device)
            V = model._extract_patch_tokens(imgs_t)
            print(f"         Patch tokens shape: {V.shape}")
            assert V.shape == (2, 257, 1024), f"Expected (2,257,1024) for ViT-L/14, got {V.shape}"

        check("CLIP patch tokens shape (B, 257, 1024)", check_patch_tokens_shape)

        def check_forward_shape_binary():
            from PIL import Image as PILImage
            imgs = [PILImage.new("RGB", (224, 224)) for _ in range(3)]
            texts = ["test caption [SEP] ocr text"] * 3
            with torch.no_grad():
                logits = model(imgs, texts)
            assert logits.shape == (3, 1), f"Expected (3,1), got {logits.shape}"

        check("Forward pass shape: binary (B, 1)", check_forward_shape_binary)

        def check_forward_shape_6class():
            from src.p2.model import TCAM
            from PIL import Image as PILImage
            m6 = TCAM.from_config(cfg_m, num_classes=6).to(device)
            imgs = [PILImage.new("RGB", (224, 224)) for _ in range(2)]
            texts = ["test text"] * 2
            with torch.no_grad():
                logits = m6(imgs, texts)
            assert logits.shape == (2, 6), f"Expected (2,6), got {logits.shape}"

        check("Forward pass shape: 6-class (B, 6)", check_forward_shape_6class)

        def check_stage2_reinit():
            import copy
            # Save proj_t weights before reinit
            proj_t_before = model.proj_t.weight.data.clone()
            cross_attn_before = model.cross_attn.attn.in_proj_weight.data.clone()

            model.reinit_for_stage2(new_num_classes=5)

            proj_t_after = model.proj_t.weight.data.clone()
            cross_attn_after = model.cross_attn.attn.in_proj_weight.data.clone()

            # proj_t MUST be unchanged
            import torch
            assert torch.allclose(proj_t_before, proj_t_after), \
                "proj_t changed after reinit_for_stage2 — should be kept!"

            # cross_attn MUST have changed
            assert not torch.allclose(cross_attn_before, cross_attn_after), \
                "cross_attn not re-initialised after reinit_for_stage2"

            # num_classes updated
            assert model.num_classes == 5, f"num_classes should be 5, got {model.num_classes}"

            # Head output dim
            assert model.head[-1].out_features == 5, \
                f"head output dim should be 5, got {model.head[-1].out_features}"

            print(f"         proj_t: KEPT (max diff={( proj_t_after - proj_t_before).abs().max():.2e})")
            print(f"         cross_attn: RESET (max diff={(cross_attn_after - cross_attn_before).abs().max():.2e})")

        check("Stage 2 reinit: proj_t kept, cross_attn/head reset", check_stage2_reinit)

    # ── 5. Loss factory ───────────────────────────────────────────────────────
    print("\n[5] Loss factory checks")

    def check_losses():
        import torch.nn as nn
        from src.p2.dataset import P2Dataset
        from src.p2.losses import get_p2_loss

        ds = P2Dataset("val", P2Config(variation="D"), stage=None, max_samples=50)

        loss_a = get_p2_loss("A", 1, ds, device)
        assert isinstance(loss_a, nn.BCEWithLogitsLoss), type(loss_a)

        loss_b = get_p2_loss("B", 1, ds, device)
        assert isinstance(loss_b, nn.CrossEntropyLoss), type(loss_b)

        loss_cs1 = get_p2_loss("C", 1, ds, device)
        assert isinstance(loss_cs1, nn.BCEWithLogitsLoss), type(loss_cs1)

        ds_s2 = P2Dataset("val", P2Config(variation="D"), stage=2, max_samples=50)
        loss_ds2 = get_p2_loss("D", 2, ds_s2, device)
        assert isinstance(loss_ds2, nn.BCEWithLogitsLoss), type(loss_ds2)

    check("Loss factory returns correct type for each variation", check_losses)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_pass = sum(1 for r, _ in results if r == PASS)
    n_fail = sum(1 for r, _ in results if r == FAIL)
    n_skip = sum(1 for r, _ in results if r == SKIP)
    total  = len(results)

    print(f"\n{'='*50}")
    print(f"Results: {n_pass}/{total} passed  |  {n_fail} failed  |  {n_skip} skipped")

    if n_skip > 0:
        print(f"  {n_skip} checks skipped (CLIP not installed — expected in local env)")

    if n_fail > 0:
        print("\nFailed checks:")
        for r, name in results:
            if r == FAIL:
                print(f"  - {name}")
        print("\nFix the failures above before training.")
        sys.exit(1)
    else:
        if n_skip > 0:
            print("Non-environment checks passed. Install CLIP on Kaggle to run model checks.")
        else:
            print("All checks passed. P2 is ready to train.")
        sys.exit(0)


if __name__ == "__main__":
    main()
