#!/usr/bin/env python3
"""Route Pluto cases through the user's CURRENT benchmark without editing it.

All remaining command-line options are forwarded to that benchmark. Use its
existing Long100-capable version, which was used for the current study.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__,add_help=False)
    ap.add_argument('--benchmark',type=Path,required=True)
    ap.add_argument('--cases-dir',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=15)
    ap.add_argument('--check-only',action='store_true')
    own,forward=ap.parse_known_args()
    if not 1<=own.limit<=15:ap.error('--limit must be 1..15')
    if any(x=='--dataset' or x.startswith('--dataset=') for x in forward):
        ap.error('This adapter selects Pluto; do not pass --dataset.')
    path=own.benchmark.expanduser().resolve(); root=own.cases_dir.expanduser().resolve()
    if not path.is_file():raise FileNotFoundError(path)
    inventory=json.loads((root/'fd_cases_manifest.json').read_text())
    items=inventory['cases'][:own.limit]
    if len(items)!=own.limit:raise RuntimeError('Unexpected case inventory')
    cases=[]
    for item in items:
        casefile=root/(item['case']+'.npz')
        if not casefile.is_file():raise FileNotFoundError(f'Generate all selected FD cases first: {casefile}')
        if sha(casefile)!=item.get('sha256'):raise RuntimeError(f'Case hash mismatch: {casefile}')
        with np.load(casefile,allow_pickle=False) as d:
            if d['vp'].shape!=(128,128) or d['snaps'].shape!=(101,128,128) or not np.isfinite(d['snaps']).all():
                raise ValueError(f'Invalid benchmark case: {casefile}')
            if json.loads(str(d['signature_json'].item()))!=item['signature'] or int(d['loc_x'])!=item['loc_x']:
                raise ValueError(f'FD signature/source location mismatch: {casefile}')
            if str(d['velocity_unit'].item())!='m/s' or not np.isfinite(d['vp']).all():
                raise ValueError(f'Invalid velocity units/data: {casefile}')
        cases.append(('pluto',item['case'],casefile))
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('_pluto_existing_benchmark',path)
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    for name in ('get_args','discover_cases','load_case','main'):
        if not callable(getattr(bench,name,None)):raise RuntimeError(f'Current benchmark does not expose {name}; use the existing common benchmark entrypoint.')
    for _,_,casefile in cases:bench.load_case(casefile)
    if own.check_only:
        print(json.dumps({'checked_cases':len(cases),'benchmark':str(path),'benchmark_sha256':sha(path),
            'adapter':'Only case discovery and dataset label are changed; use current benchmark loaders/predictors/metrics/timers.'},indent=2))
        return
    if not any(x=='--seed' or x.startswith('--seed=') for x in forward):forward.extend(['--seed','5100'])
    original_get_args=bench.get_args
    resolved={}
    def get_args_pluto():
        args=original_get_args()
        args.dataset=['pluto']
        resolved['args']=args
        return args
    bench.get_args=get_args_pluto
    bench.discover_cases=lambda repo,selected:cases
    sys.argv=[str(path)]+forward
    bench.main()
    args=resolved['args'];out=Path(args.out_dir).expanduser().resolve()
    record={'dataset':'pluto','case_count':len(cases),'benchmark_path':str(path),'benchmark_sha256':sha(path),
        'requested_seed':args.seed,'case_inventory':items,
        'source_inventory_sha256':sha(root/'fd_cases_manifest.json'),
        'note':'Benchmark source file was not modified. Common eager evaluation and existing model weights are reused.'}
    (out/'pluto_dataset_provenance.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Pluto provenance saved to {out}',flush=True)

if __name__=='__main__':main()
