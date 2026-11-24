#!/usr/bin/env python3
"""Check physical units, crop inventory, checksums and SEG-Y/NPY equivalence."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    m=json.loads((ROOT/'manifest.json').read_text())
    full=np.load(ROOT/'full/Pluto_full_vp_mps.npy',allow_pickle=False)
    assert full.shape==(1201,6960) and full.dtype==np.float32 and np.isfinite(full).all()
    assert float(full.min())==m['source']['vp_min_mps'] and float(full.max())==m['source']['vp_max_mps']
    raw=(ROOT/'full/Pluto_full_vp_mps.sgy').read_bytes()
    records=np.ndarray((6960,),dtype=np.dtype([('header','u1',(240,)),('values','>f4',(1201,))]),buffer=raw,offset=3600)
    assert np.array_equal(records['values'].astype(np.float32).T,full)
    cases=0
    for model in m['models']:
        path=ROOT/model['path'];assert sha(path)==model['sha256']
        with np.load(path,allow_pickle=False) as d:
            a=d['vp'];assert a.shape==(128,128) and a.dtype==np.float32 and np.isfinite(a).all()
            assert str(d['velocity_unit'].item())=='m/s'
            assert 1450<=a.min()<=a.max()<=5000
            assert np.array_equal(a,np.load(path.with_suffix('.npy'),allow_pickle=False))
        assert all(0<=x<128 for x in model['shots_x']) and len(set(model['shots_x']))==5
        cases+=len(model['shots_x'])
        print(f"{model['name']}: {a.shape}, {float(a.min()):.3f}–{float(a.max()):.3f} m/s")
    assert cases==15
    checksums=ROOT/'SHA256SUMS.txt'
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            digest,relative=line.split('  ',1)
            assert sha(ROOT/relative)==digest,f'Checksum mismatch: {relative}'
    print('PASS: m/s conversion, orientation, 3 velocity models, 15 planned cases, no clipping or normalization.')
if __name__=='__main__':main()
