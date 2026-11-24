#!/usr/bin/env python3
"""Generate 15 FD reference cases using the existing GenWP FD10 operators."""
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, os
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--limit',type=int,default=15,help='Use 1 for the first-case smoke run; rerun with 15 to complete.')
    ap.add_argument('--verify-case',type=Path,help='Optional existing NPZ/MAT reference to recompute before new cases.')
    args=ap.parse_args()
    if not 1<=args.limit<=15:ap.error('--limit must be 1..15')
    repo=args.repo.expanduser().resolve(); out=args.out_dir.expanduser().resolve()
    fd_source=repo/'dataset/data_gen/fd_torch.py'
    if not fd_source.is_file():raise FileNotFoundError(fd_source)
    import torch
    from fd_reference import fd_predict
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('Run this in the existing CUDA/MCWP environment on the server.')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    manifest=json.loads((ROOT/'manifest.json').read_text())
    fd=manifest['fd']
    cfg=SimpleNamespace(fd_dx=fd['dx_m'],fd_dz=fd['dz_m'],fd_dt=fd['dt_s'],
        fd_nt=fd['nt'],fd_freq=fd['ricker_hz'],fd_pml=fd['pml_cells'],fd_downsample=fd['time_downsample'])
    out.mkdir(parents=True,exist_ok=True)
    code_hash=sha(fd_source)
    if args.verify_case:
        path=args.verify_case.expanduser().resolve()
        if path.suffix=='.npz':
            with np.load(path,allow_pickle=False) as d:
                vp=np.asarray(d['vp'],np.float32);ref=np.asarray(d['snaps'],np.float32);loc=int(d['loc_x'])
        elif path.suffix=='.mat':
            from scipy.io import loadmat
            d=loadmat(path);vp=np.asarray(d['vp'],np.float32);ref=np.asarray(d['snaps'],np.float32);loc=int(np.asarray(d['loc_x']).item())
        else:raise ValueError('Verification case must be a stored NPZ or MAT with vp/snaps/loc_x.')
        if ref.shape==(501,128,128):ref=ref[::5]
        recomputed=fd_predict(repo,vp,loc,device,cfg).cpu().numpy()
        if ref.shape!=recomputed.shape:raise ValueError(f'Verification shape mismatch: {ref.shape}, {recomputed.shape}')
        error=recomputed.astype(np.float64)-ref.astype(np.float64)
        print('Existing-case FD check:',json.dumps({'case':str(path),'mae':float(abs(error).mean()),'max_abs':float(abs(error).max())}),flush=True)
        if not np.allclose(recomputed,ref,rtol=1e-5,atol=1e-5):
            raise RuntimeError('Existing-case FD check failed. Resolve the solver/source/grid settings before generating new cases.')
    planned=[]; ordinal=0
    for model in manifest['models']:
        velocity_path=ROOT/model['path']
        if sha(velocity_path)!=model['sha256']:raise RuntimeError(f'Input hash mismatch: {velocity_path}')
        with np.load(velocity_path,allow_pickle=False) as d:vp=np.asarray(d['vp'],np.float32)
        if vp.shape!=(128,128) or not np.isfinite(vp).all() or vp.min()<1450 or vp.max()>5000:
            raise ValueError('Invalid physical velocity input; do not normalize the saved array.')
        for shot,loc_x in enumerate(model['shots_x'],1):
            case=f"Pluto_vel{model['model_id']}_shot{shot}"
            global_index=manifest['case_index_start']+ordinal
            path=out/(case+'.npz')
            signature={'fd':fd,'velocity_sha256':model['sha256'],'fd_torch_sha256':code_hash,
                'snapshot_loop_sha256':sha(ROOT/'fd_reference.py'),'loc_x':loc_x}
            item={'dataset':'pluto','case':case,'path':str(path),'velocity_model':model['name'],
                'model_id':model['model_id'],'shot':shot,'loc_x':loc_x,'global_case_index':global_index,
                'genwp_seed':100*global_index,'signature':signature}
            planned.append(item)
            ordinal+=1
            if ordinal>args.limit:continue
            if path.exists():
                with np.load(path,allow_pickle=False) as d:
                    old=json.loads(str(d['signature_json'].item()))
                    if old!=signature or d['snaps'].shape!=(101,128,128) or not np.array_equal(d['vp'],vp) or not np.isfinite(d['snaps']).all():
                        raise RuntimeError(f'Existing case is incompatible; kept unchanged: {path}')
                print(f'[{ordinal}/{args.limit}] verified existing {case}',flush=True)
                continue
            print(f'[{ordinal}/{args.limit}] FD10 {case} · source x={loc_x}',flush=True)
            snaps=fd_predict(repo,vp,loc_x,device,cfg).cpu().numpy().astype(np.float32)
            if snaps.shape!=(101,128,128) or not np.isfinite(snaps).all() or not np.any(snaps):
                raise RuntimeError(f'Invalid FD wavefield: {case}')
            temporary=path.with_suffix('.npz.writing')
            with temporary.open('wb') as handle:
                np.savez_compressed(handle,vp=vp,snaps=snaps,loc_x=np.int64(loc_x),
                    time_s=np.arange(101,dtype=np.float32)*np.float32(.01),
                    global_case_index=np.int64(global_index),genwp_seed=np.int64(100*global_index),
                    velocity_unit=np.asarray('m/s'),signature_json=np.asarray(json.dumps(signature,sort_keys=True)),
                    source_dataset=np.asarray('Pluto 1.5'),source_sha256=np.asarray(manifest['source']['source_sha256']))
            os.replace(temporary,path)
            with np.load(path,allow_pickle=False) as d:
                assert d['vp'].shape==(128,128) and d['snaps'].shape==(101,128,128) and int(d['loc_x'])==loc_x
            print(f"  saved {path.name} · max |pressure|={float(abs(snaps).max()):.6g}",flush=True)
    for item in planned:
        path=Path(item['path']);item['exists']=path.exists()
        if path.exists():item['sha256']=sha(path)
    inventory={'dataset':'pluto','expected_case_count':15,'created_or_existing_cases':sum(x['exists'] for x in planned),
        'source_manifest':manifest,'gpu':torch.cuda.get_device_name(device),'precision':'FP32 / TF32 off',
        'cases':planned}
    (out/'fd_cases_manifest.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print(f"Ready: {inventory['created_or_existing_cases']}/15 cases in {out}",flush=True)

if __name__=='__main__':main()
