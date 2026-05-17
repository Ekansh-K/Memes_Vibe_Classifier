import sys
sys.path.insert(0, '.')
from src.data.splits import load_split_ids, load_processed_labels

labels = load_processed_labels()
rows = []
total = 0
for split in ['train', 'val', 'test']:
    ids  = load_split_ids(split)
    hate = sum(1 for i in ids if i in labels and labels[i]['hard_label_binary'] == 1)
    rows.append({'split': split, 'n': len(ids), 'hate': hate})
    total += len(ids)

print("Split   Count      pct    Hate   Hate%")
print("-"*45)
for r in rows:
    print(r['split'].ljust(6), str(r['n']).rjust(8),
          (str(round(r['n']/total*100,1))+'%').rjust(7),
          str(r['hate']).rjust(7),
          (str(round(r['hate']/r['n']*100,1))+'%').rjust(7))
print("-"*45)
print("TOTAL  ", total)

hate_rates = [r['hate']/r['n'] for r in rows]
diff = max(hate_rates) - min(hate_rates)
print("Hate rate variance:", round(diff*100,2), "%  --", "OK" if diff < 0.02 else "MISMATCH")

train_set = set(load_split_ids('train'))
val_set   = set(load_split_ids('val'))
test_set  = set(load_split_ids('test'))
no_overlap = len(train_set & val_set)==0 and len(train_set & test_set)==0 and len(val_set & test_set)==0
print("No overlap:", no_overlap)
