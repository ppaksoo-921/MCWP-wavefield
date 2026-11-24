"""Optional plots, one separate figure per case; matplotlib is optional."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('csvs', nargs='+')
p.add_argument('--out-dir', required=True)
args = p.parse_args()
rows = []
for path in args.csvs:
    with open(path) as f:
        rows.extend(csv.DictReader(f))
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
for case in ('Overthrust','SEGEAGE','Marmousi'):
    groups = defaultdict(list)
    officials = []
    for r in rows:
        if r['case'] != case or r['status'] != 'PASS':
            continue
        if r['label'] == 'authors_final':
            officials.append(float(r['mae_all101']))
        elif r['steps']:
            name = r['label'].rsplit('_',1)[0]
            groups[(name,int(r['steps']))].append(float(r['mae_all101']))
    fig, ax = plt.subplots(figsize=(7,4.5))
    for name in sorted({key[0] for key in groups}):
        points = sorted((step,sum(values)/len(values)) for (kind,step),values in groups.items() if kind==name)
        ax.plot([x for x,y in points],[y for x,y in points],marker='o',label=name)
    if officials:
        ax.axhline(sum(officials)/len(officials),linestyle='--',label='Authors final GenWP')
    ax.set(xlabel='Completed optimizer updates',ylabel='MAE (101 frames; shot/seed mean)',title=case)
    ax.legend(); fig.tight_layout(); fig.savefig(out/f'{case}_mae.png',dpi=180); plt.close(fig)
print(out.resolve())
