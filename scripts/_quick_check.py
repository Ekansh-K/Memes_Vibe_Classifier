import json
nb = json.load(open('notebooks/p2_tcam_train.ipynb', encoding='utf-8'))
for i, c in enumerate(nb['cells'][:5]):
    src = ''.join(c.get('source', []))
    has_soft = 'SOFT_LABELS' in src
    ctype = c['cell_type']
    print(f'Cell {i} ({ctype}): SOFT_LABELS={has_soft}  len={len(src)}')
    if has_soft:
        for k in ['SOFT_LABELS', 'AGREEMENT_WEIGHTING', 'LABEL_SMOOTHING', 'TEMP_SCALING']:
            print(f'  {k}: {k in src}')
        break

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if 'P2Config(' in src:
        for k in ['use_soft_labels', 'use_agreement_weighting', 'agreement_weights', 'temperature_scaling']:
            print(f'  P2Config.{k}: {k in src}')
        break
print('Done')
