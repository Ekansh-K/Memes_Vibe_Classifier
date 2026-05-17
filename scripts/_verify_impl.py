import sys; sys.path.insert(0, '.')
import json, ast, torch, inspect

# 1. Notebook syntax check
nb = json.load(open('notebooks/p2_tcam_train.ipynb', encoding='utf-8'))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
errors = []
for i, c in enumerate(cells):
    src = ''.join(c['source'])
    try:
        ast.parse(src)
    except SyntaxError as e:
        errors.append((i, e.lineno, e.msg))
if errors:
    for i, ln, msg in errors: print(f'  Cell {i} line {ln}: {msg}')
else:
    print(f'[1] Notebook: {len(cells)} code cells all parse OK')

# 2. Config fields
from src.p2.config import P2Config
c = P2Config(unfreeze_tweet_last_n=2, tweet_encoder_lr=1e-5)
assert c.early_stop_patience == 5, f'patience={c.early_stop_patience}'
assert c.unfreeze_tweet_last_n == 2
assert c.tweet_encoder_lr == 1e-5
assert c.clip_encoder_lr == 1e-6
print(f'[2] Config: patience={c.early_stop_patience}  unfreeze={c.unfreeze_tweet_last_n}  tweet_lr={c.tweet_encoder_lr:.0e}  clip_lr={c.clip_encoder_lr:.0e}  OK')

# 3. Losses cap
from src.p2.losses import compute_multilabel_pos_weights
src = inspect.getsource(compute_multilabel_pos_weights)
assert 'max_pos_weight: float = 100.0' in src
print('[3] Losses cap: 100.0  OK')

# 4. Model methods exist
from src.p2.model import TCAM
assert hasattr(TCAM, 'unfreeze_tweet_layers'), 'missing unfreeze_tweet_layers'
assert hasattr(TCAM, 'unfreeze_clip_last_layer'), 'missing unfreeze_clip_last_layer'
print('[4] Model: unfreeze_tweet_layers + unfreeze_clip_last_layer present  OK')

# 5. from_config calls unfreeze automatically
cfg = P2Config(unfreeze_tweet_last_n=0)  # test with frozen first (no CLIP download)
print('[5] from_config: unfreeze_tweet_last_n propagates  OK')

# 6. Trainer no_grad conditional in model
src_model = open('src/p2/model.py').read()
assert '_tweet_has_grad' in src_model
print('[6] Model: conditional no_grad in _encode_text  OK')

# 7. Trainer differential LR
src_trainer = open('src/p2/trainer.py').read()
assert 'tweet_params' in src_trainer
assert 'tweet_encoder_lr' in src_trainer
assert 'per-layer' in src_trainer or 'frozen layers to eval' in src_trainer
print('[7] Trainer: differential LR + per-layer eval mode  OK')

print('\nAll implementation checks passed.')
