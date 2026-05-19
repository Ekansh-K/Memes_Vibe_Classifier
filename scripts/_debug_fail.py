import sys; sys.path.insert(0, '.'); sys.stdout.reconfigure(encoding='utf-8')
import torch
from src.p2.losses import SoftBCEWithAgreementWeighting

loss_fn = SoftBCEWithAgreementWeighting(
    pos_weight=torch.tensor([4.8]),
    agreement_weights=(0.2, 0.5, 1.0),
    use_agreement_weighting=True,
    label_smoothing=0.1,
)
aw = loss_fn._aw.tolist()
print(f'_aw values: {aw}')
print(f'Expected:   [0.2, 0.5, 1.0]')
print(f'Match: {aw == [0.2, 0.5, 1.0]}')
print(f'Individual: {[aw[0]==0.2, aw[1]==0.5, aw[2]==1.0]}')
print(f'Close: {all(abs(a-b)<0.001 for a,b in zip(aw, [0.2, 0.5, 1.0]))}')

# Check trainer source for temperature
src = open('src/p2/trainer.py', encoding='utf-8').read()
for line in src.split('\n'):
    if 'temperature' in line and 'return' in line:
        print(f'Return line: [{line.strip()}]')
print(f'Has double-quote temperature: {"temperature" in src}')
# The check was for "'temperature': temp_value" with single quotes
# But the actual code may use double quotes
has_single = "'temperature': temp_value" in src
has_double = '"temperature": temp_value' in src
print(f'Single-quote match: {has_single}')
print(f'Double-quote match: {has_double}')
