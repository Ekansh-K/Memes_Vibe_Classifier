"""Verify the corrected eval notebook has the right Qwen3 API."""
import json
from pathlib import Path

nb_path = Path(__file__).parent.parent / 'notebooks' / 'vlm_caption_eval_kaggle.ipynb'
nb = json.loads(nb_path.read_text(encoding='utf-8'))
cell = nb['cells'][6]
src = ''.join(cell['source'])

checks = [
    ('AutoModelForImageTextToText',       'correct model class'),
    ('return_video_kwargs=True',           '3-value process_vision_info call'),
    ('image_inputs, video_inputs, video_kwargs', '3-value unpack'),
    ('**video_kwargs',                     'video_kwargs forwarded to processor'),
    ('next(model.parameters()).device',    'correct device targeting'),
    ('min_pixels=256 * 256',              'min_pixels set in processor'),
    ('max_pixels=448 * 448',              'max_pixels set in processor'),
    ("device_map='auto'",                 'device_map auto for dual-GPU'),
]

print('=== Qwen3 cell API checks (cell 6) ===')
all_ok = True
for pattern, desc in checks:
    ok = pattern in src
    status = 'OK  ' if ok else 'MISS'
    print(f'  [{status}] {desc}')
    if not ok:
        all_ok = False
        print(f'         pattern not found: {pattern!r}')

print()
print('ALL CHECKS PASSED ✓' if all_ok else '⚠ SOME CHECKS FAILED')
