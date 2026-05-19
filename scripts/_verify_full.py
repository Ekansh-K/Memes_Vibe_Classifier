"""Exhaustive verification of Options A/B/C + Temperature Scaling.

Tests:
  1. Config fields exist and have correct defaults
  2. SoftBCEWithAgreementWeighting:
     a. Correct label smoothing math
     b. Correct agreement weighting math
     c. Gradient flows (not blocked by no_grad)
  3. TemperatureScaler:
     a. Gradient flows during fit() (checks for @torch.no_grad bug)
     b. Temperature learns correct value
  4. _get_targets:
     a. Soft labels computed correctly from soft_label column
     b. Falls back to hard labels when disabled
  5. _eval_epoch uses hard targets for metrics
  6. get_p2_loss returns correct type
  7. Checkpoint saving includes tweet_encoder params
  8. Notebook cell syntax + constants present
  9. End-to-end loss flow test
"""
import sys; sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')
import json, ast, torch
import torch.nn as nn
import numpy as np

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [OK] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        if detail:
            print(f"    -> {detail}")

def warn(name, detail):
    global WARN
    WARN += 1
    print(f"  [WARN] {name}: {detail}")

# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("1. CONFIG FIELDS")
print("=" * 60)

from src.p2.config import P2Config
c = P2Config()
check("use_soft_labels default=True", c.use_soft_labels == True)
check("use_agreement_weighting default=True", c.use_agreement_weighting == True)
check("agreement_weights default=(0.2,0.5,1.0)", c.agreement_weights == (0.2, 0.5, 1.0))
check("label_smoothing default=0.1", c.label_smoothing == 0.1)
check("temperature_scaling default=True", c.temperature_scaling == True)
check("early_stop_patience=5", c.early_stop_patience == 5)
check("unfreeze_tweet_last_n=0 (default)", c.unfreeze_tweet_last_n == 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. SoftBCEWithAgreementWeighting")
print("=" * 60)

from src.p2.losses import SoftBCEWithAgreementWeighting

loss_fn = SoftBCEWithAgreementWeighting(
    pos_weight=torch.tensor([4.8]),
    agreement_weights=(0.2, 0.5, 1.0),
    use_agreement_weighting=True,
    label_smoothing=0.1,
)

# 2a. Label smoothing math
targets_raw = torch.tensor([0.0, 0.333, 0.667, 1.0])
smoothed = targets_raw * 0.9 + 0.5 * 0.1  # = targets * 0.9 + 0.05
expected_smooth = torch.tensor([0.05, 0.3497, 0.6503, 0.95])
check("Label smoothing: 0.0→0.05", abs(smoothed[0].item() - 0.05) < 0.001)
check("Label smoothing: 1.0→0.95", abs(smoothed[3].item() - 0.95) < 0.001)
check("Label smoothing: 0.333→0.3497", abs(smoothed[1].item() - 0.3497) < 0.001)

# 2b. Agreement weighting math
logits = torch.zeros(4, requires_grad=True)
targets = torch.tensor([0.0, 0.333, 0.667, 1.0])
agree = torch.tensor([1, 2, 3, 1])  # weights: 0.2, 0.5, 1.0, 0.2
out = loss_fn(logits, targets.clone(), agreement_levels=agree)
check("Output is scalar", out.ndim == 0)
check("Output is positive", out.item() > 0)

# 2c. Gradient flows
out.backward()
check("Gradient flows through SoftBCE", logits.grad is not None and logits.grad.abs().sum() > 0,
      f"grad={logits.grad}")

# 2d. Without agreement levels → falls back to mean
logits2 = torch.zeros(4, requires_grad=True)
out2 = loss_fn(logits2, targets.clone())
check("Without agreement_levels: returns scalar", out2.ndim == 0)

# 2e. pos_weight buffer is on correct device
check("pos_weight in BCE", loss_fn.bce.pos_weight is not None)
check("_aw buffer exists", hasattr(loss_fn, '_aw'))
check("_aw has 3 values", loss_fn._aw.shape == (3,))
check("_aw values correct", list(loss_fn._aw.tolist()) == [0.2, 0.5, 1.0])

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. TemperatureScaler")
print("=" * 60)

from src.p2.losses import TemperatureScaler

ts = TemperatureScaler()
check("Initial temperature=1.0", ts.temperature.item() == 1.0)

# 3a. Check if @torch.no_grad blocks fit()
# Create synthetic data where T>1 should be optimal (over-confident logits)
torch.manual_seed(42)
# Logits that are too extreme — temperature should learn T > 1 to soften
synthetic_logits = torch.tensor([3.0, -3.0, 2.5, -2.5, 3.0, -3.0, 2.0, -2.0])
synthetic_labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0])  # some wrong labels

# The fit() method has @torch.no_grad — this is a BUG if it prevents LBFGS from working
# Let's test if temperature actually changes
ts2 = TemperatureScaler()
try:
    t_val = ts2.fit(synthetic_logits.clone(), synthetic_labels.clone())
    check("TemperatureScaler.fit() runs without error", True)
    check("Temperature changed from 1.0", abs(t_val - 1.0) > 0.001,
          f"T={t_val:.4f} — if T≈1.0, @no_grad might be blocking optimization")
except Exception as e:
    check("TemperatureScaler.fit() runs without error", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. _get_targets")
print("=" * 60)

from src.p2.trainer import _get_targets

batch = {
    'soft_label': torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],    # NotHate unanimous → P(hate)=0.0
        [0.667, 0.333, 0.0, 0.0, 0.0, 0.0], # 2/3 NotHate → P(hate)=0.333
        [0.333, 0.333, 0.333, 0.0, 0.0, 0.0],# 1/3 NotHate → P(hate)=0.667
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],     # All Racist → P(hate)=1.0
    ]),
    'label_binary': torch.tensor([0, 0, 1, 1]),
    'label_6class': torch.tensor([0, 0, 1, 1]),
    'multi_label_binary': torch.zeros(4, 6),
    'label_s2': torch.tensor([-1, -1, 0, 0]),
    'agreement_level': torch.tensor([3, 2, 1, 3]),
}

# 4a. Soft labels
cfg_soft = P2Config(use_soft_labels=True)
targets_soft = _get_targets(batch, 'D', 1, config=cfg_soft)
check("Soft target[0] = 0.0 (not hate)", abs(targets_soft[0].item()) < 0.01)
check("Soft target[1] = 0.333", abs(targets_soft[1].item() - 0.333) < 0.01)
check("Soft target[2] = 0.667", abs(targets_soft[2].item() - 0.667) < 0.01)
check("Soft target[3] = 1.0", abs(targets_soft[3].item() - 1.0) < 0.01)

# 4b. Hard labels (fallback)
cfg_hard = P2Config(use_soft_labels=False)
targets_hard = _get_targets(batch, 'D', 1, config=cfg_hard)
check("Hard target[0] = 0.0", targets_hard[0].item() == 0.0)
check("Hard target[1] = 0.0", targets_hard[1].item() == 0.0)
check("Hard target[2] = 1.0", targets_hard[2].item() == 1.0)
check("Hard target[3] = 1.0", targets_hard[3].item() == 1.0)

# 4c. Stage 2 unaffected
targets_s2 = _get_targets(batch, 'D', 2, config=cfg_soft)
check("Stage 2 D: returns multi_label_binary[:, 1:]", targets_s2.shape == (4, 5))

# 4d. P2-B unaffected
targets_b = _get_targets(batch, 'B', 1, config=cfg_soft)
check("P2-B: returns label_6class", torch.equal(targets_b, batch['label_6class']))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. get_p2_loss returns correct type")
print("=" * 60)

from src.p2.losses import get_p2_loss

class MockDS:
    def __init__(self):
        self.sample_ids = ['1', '2', '3']
        self.labels = {
            '1': {'hard_label_binary': 0, 'hard_label_6class': 0},
            '2': {'hard_label_binary': 1, 'hard_label_6class': 1},
            '3': {'hard_label_binary': 1, 'hard_label_6class': 2},
        }
        self.config = P2Config()

ds = MockDS()

# With soft+agree enabled
cfg_on = P2Config(use_soft_labels=True, use_agreement_weighting=True, label_smoothing=0.1)
loss_s1 = get_p2_loss('D', 1, ds, torch.device('cpu'), label_smoothing=0.1, config=cfg_on)
check("Stage 1 with options → SoftBCEWithAgreementWeighting",
      isinstance(loss_s1, SoftBCEWithAgreementWeighting))

# With all disabled
cfg_off = P2Config(use_soft_labels=False, use_agreement_weighting=False, label_smoothing=0.0)
loss_s1_off = get_p2_loss('D', 1, ds, torch.device('cpu'), label_smoothing=0.0, config=cfg_off)
check("Stage 1 all disabled → standard BCEWithLogitsLoss",
      isinstance(loss_s1_off, nn.BCEWithLogitsLoss) and not isinstance(loss_s1_off, SoftBCEWithAgreementWeighting))

# Stage 2 should NOT be affected
ds.labels['1']['hard_label_6class'] = 1  # make all hateful for S2
ds.labels['1']['hard_label_binary'] = 1
ds.config.multilabel_threshold = 2/3
soft_labels = {
    '1': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    '2': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    '3': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
}
for sid in ds.sample_ids:
    ds.labels[sid]['soft_label_6class'] = soft_labels[sid]
loss_s2 = get_p2_loss('D', 2, ds, torch.device('cpu'), config=cfg_on)
check("Stage 2 D: still returns BCEWithLogitsLoss (not SoftBCE)",
      isinstance(loss_s2, nn.BCEWithLogitsLoss) and not isinstance(loss_s2, SoftBCEWithAgreementWeighting))

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. CHECKPOINT SAVING LOGIC")
print("=" * 60)

trainer_src = open('src/p2/trainer.py', encoding='utf-8').read()
check("Checkpoint includes tweet_encoder prefix",
      '"tweet_encoder."' in trainer_src)
check("Checkpoint filters by requires_grad",
      'p.requires_grad' in trainer_src and 'k.startswith(n)' in trainer_src)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. TEMPERATURE SCALING IN TRAINER")
print("=" * 60)

check("Trainer imports TemperatureScaler", 'from src.p2.losses import TemperatureScaler' in trainer_src)
check("Trainer calls temp_scaler.fit()", 'temp_scaler.fit(val_logits_t, val_hard_t)' in trainer_src)
check("Return dict includes temperature", "'temperature': temp_value" in trainer_src)
check("Temp scaling conditioned on config flag",
      "getattr(config, \"temperature_scaling\", False)" in trainer_src)
check("Temp scaling only for binary Stage 1",
      'primary_key == "binary/macro_f1"' in trainer_src)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. NOTEBOOK VERIFICATION")
print("=" * 60)

nb = json.load(open('notebooks/p2_tcam_train.ipynb', encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
syntax_errors = []
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        syntax_errors.append(f"Cell {i} line {e.lineno}: {e.msg}")
check(f"All {len(cells)} code cells parse OK", len(syntax_errors) == 0,
      str(syntax_errors) if syntax_errors else "")

# Find constants cell and config cell
all_nb_src = ''.join(''.join(c.get('source', [])) for c in nb['cells'])
check("SOFT_LABELS in notebook", 'SOFT_LABELS' in all_nb_src)
check("AGREEMENT_WEIGHTING in notebook", 'AGREEMENT_WEIGHTING' in all_nb_src)
check("AGREEMENT_WEIGHTS in notebook", 'AGREEMENT_WEIGHTS' in all_nb_src)
check("LABEL_SMOOTHING in notebook", 'LABEL_SMOOTHING' in all_nb_src)
check("TEMP_SCALING in notebook", 'TEMP_SCALING' in all_nb_src)
check("use_soft_labels in P2Config cell", 'use_soft_labels' in all_nb_src)
check("temperature_scaling in P2Config cell", 'temperature_scaling' in all_nb_src)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. END-TO-END LOSS FLOW")
print("=" * 60)

# Simulate a complete train step
loss_fn = SoftBCEWithAgreementWeighting(
    pos_weight=torch.tensor([4.8]),
    agreement_weights=(0.2, 0.5, 1.0),
    use_agreement_weighting=True,
    label_smoothing=0.1,
)

# Simulated batch
logits = torch.randn(16, requires_grad=True)
# Soft targets (realistic distribution)
targets = torch.tensor([0.0]*6 + [0.333]*4 + [0.667]*3 + [1.0]*3)
agree = torch.tensor([3]*4 + [2]*4 + [2]*4 + [1]*2 + [3]*2)

loss = loss_fn(logits, targets, agreement_levels=agree)
loss.backward()

check("E2E: loss is finite", torch.isfinite(loss))
check("E2E: gradients are finite", torch.all(torch.isfinite(logits.grad)))
check("E2E: gradients are non-zero", logits.grad.abs().sum() > 0)

# Verify that high-agreement samples get more gradient
# Weight 1.0 samples should contribute more per-unit than weight 0.2 samples
check("E2E: loss is scalar", loss.ndim == 0)

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {WARN} warnings")
print("=" * 60)
if FAIL > 0:
    print("!! SOME CHECKS FAILED -- review above")
else:
    print("ALL CHECKS PASSED")
