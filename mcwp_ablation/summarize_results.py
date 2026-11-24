"""Summarize evaluator CSV without pandas. Two shots are not independent geologies."""
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('metrics_csv')
args=p.parse_args()
source=Path(args.metrics_csv).expanduser().resolve()
with open(source, newline='') as f:
    rows=list(csv.DictReader(f))
groups=defaultdict(list)
for row in rows:
    groups[(row['label'],row['case'])].append(row)
out=[]
for (label,case), group in groups.items():
    good=[r for r in group if r['status']=='PASS' and math.isfinite(float(r['mae_pred100']))]
    item={'label':label,'case':case,'n_runs':len(group),'n_valid':len(good),'n_failed':len(group)-len(good)}
    for key in ['mae_all101','mae_pred100','relative_l2_pred100','snr_db_pred100','runtime_sec']:
        values=[float(r[key]) for r in good]
        item[key+'_mean']=statistics.mean(values) if values else float('nan')
    item['parameters']=group[0]['parameters']
    out.append(item)
result=source.parent/'summary_by_case.csv'
with open(result,'w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(out[0]) if out else ['label','case'])
    writer.writeheader(); writer.writerows(out)
print('label                            case          MAE(pred100)   relL2      SNR(dB)  failed')
for row in out:
    print(f"{row['label']:32s} {row['case']:12s} {row['mae_pred100_mean']:12.6g} {row['relative_l2_pred100_mean']:9.4g} {row['snr_db_pred100_mean']:9.3f} {row['n_failed']:4d}")
print('Saved:',result)
print('Means describe these test cases only. Inference seeds are NOT independent training seeds.')
