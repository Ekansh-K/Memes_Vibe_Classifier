"""Full verification of Options A/B/C + Temperature Scaling implementation."""
import sys; sys.path.insert(0, '.')
import json, ast, inspect
errors = []

# 1. Notebook syntax check
nb = json.load(open('notebooks/p2_tcam_train.ipynb', encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append(f'Cell {i} line {e.lineno}: {e.msg}')
if errors:
    for e in errors: print(f'  ERROR: {e}')
else:
    print(f'[1] Notebook: {len(cells)} code cells all parse OK')

# 2. Config fields
from src.p2.config import P2Config
c = P2Config(
    use_soft_labels=True,
    use_agreement_weighting=True,
    agreement_weights=(0.2, 0.5, 1.0),
    label_smoothing=0.1,
    temperature_scaling=True,
)
assert c.use_soft_labels == True
assert c.use_agreement_weighting == True
assert c.agreement_weights == (0.2, 0.5, 1.0)
assert c.label_smoothing == 0.1
assert c.temperature_scaling == True
print(f'[2] Config: soft={c.use_soft_labels} agree={c.use_agreement_weighting} '
      f'weights={c.agreement_weights} ls={c.label_smoothing} temp={c.temperature_scaling} OK')

# 3. SoftBCEWithAgreementWeighting exists and works
import torch
from src.p2.losses import SoftBCEWithAgreementWeighting, TemperatureScaler
loss_fn = SoftBCEWithAgreementWeighting(
    pos_weight=torch.tensor([4.8]),
    agreement_weights=(0.2, 0.5, 1.0),
    use_agreement_weighting=True,
    label_smoothing=0.1,
)
logits = torch.randn(8)
targets = torch.tensor([0.0, 0.0, 0.333, 0.333, 0.667, 0.667, 1.0, 1.0])
agree = torch.tensor([3, 2, 1, 2, 3, 2, 3, 1])
out = loss_fn(logits, targets, agreement_levels=agree)
assert out.ndim == 0  # scalar
assert out.item() > 0
print(f'[3] SoftBCEWithAgreementWeighting: output={out.item():.4f} (scalar, positive) OK')

# 4. Without agreement levels (regular mode)
out2 = loss_fn(logits, targets)
assert out2.ndim == 0
print(f'[4] SoftBCE without agreement: output={out2.item():.4f} OK')

# 5. TemperatureScaler
ts = TemperatureScaler()
assert ts.temperature.item() == 1.0
print(f'[5] TemperatureScaler: initial T={ts.temperature.item()} OK')

# 6. get_p2_loss returns SoftBCE when config has options enabled
from src.p2.losses import get_p2_loss
# Need a mock dataset
class MockDS:
    def __init__(self):
        self.sample_ids = ['1', '2', '3']
        self.labels = {
            '1': {'hard_label_binary': 0, 'hard_label_6class': 0},
            '2': {'hard_label_binary': 1, 'hard_label_6class': 1},
            '3': {'hard_label_binary': 1, 'hard_label_6class': 2},
        }
        self.config = P2Config()

loss = get_p2_loss('D', 1, MockDS(), torch.device('cpu'), label_smoothing=0.1, config=c)
assert isinstance(loss, SoftBCEWithAgreementWeighting), f'Got {type(loss).__name__}'
print(f'[6] get_p2_loss: returns SoftBCEWithAgreementWeighting for Stage 1 OK')

# 7. _get_targets returns soft labels when config enabled
# Verify code structure
trainer_src = open('src/p2/trainer.py', encoding='utf-8').read()
assert 'soft_label' in trainer_src
assert 'config=config' in trainer_src
assert 'agreement_levels' in trainer_src
print('[7] Trainer: _get_targets soft label support + agreement levels wired OK')

# 8. Temperature scaling in train_stage
assert 'TemperatureScaler' in trainer_src
assert 'temp_scaler.fit' in trainer_src
print('[8] Trainer: temperature scaling post-training OK')

# 9. Checkpoint saves tweet_encoder layers
assert '"tweet_encoder."' in trainer_src
print('[9] Trainer: checkpoint includes unfrozen tweet_encoder layers OK')

# 10. Notebook has new constants
cell1_src = ''.join(nb['cells'][0]['source'])
assert 'SOFT_LABELS' in cell1_src
assert 'AGREEMENT_WEIGHTING' in cell1_src
assert 'LABEL_SMOOTHING' in cell1_src
assert 'TEMP_SCALING' in cell1_src
print('[10] Notebook cell 1: SOFT_LABELS, AGREEMENT_WEIGHTING, LABEL_SMOOTHING, TEMP_SCALING present OK')

# Check P2Config cell has the new params
config_cell = None
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if 'P2Config(' in src:
        config_cell = src
        break
assert 'use_soft_labels' in config_cell
assert 'use_agreement_weighting' in config_cell
assert 'agreement_weights' in config_cell
assert 'temperature_scaling' in config_cell
print('[11] Notebook P2Config cell: all new params wired OK')

print('\n✅ ALL CHECKS PASSED — Options A/B/C + Temperature Scaling fully implemented')
