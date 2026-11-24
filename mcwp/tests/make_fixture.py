"""Synthetic SOFTWARE-TEST fixtures. These are NOT FD reference wavefields."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat

p = argparse.ArgumentParser()
p.add_argument('root')
p.add_argument('--files', type=int, default=12)
args = p.parse_args()
root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
z,x = np.mgrid[:128,:128].astype(np.float32)
t = np.arange(101, dtype=np.float32)[:,None,None]
for i in range(args.files):
    vp = (2200 + 5*z + 2*x + i).astype(np.float32)
    a = np.exp(-((z[None]-(10+t*.8))**2+(x[None]-(35+i))**2)/100)
    u = (.15*a*np.cos(.25*t)).astype(np.float32)
    with h5py.File(root/f'synthetic_vel{i+1}_shot1.h5','w') as f:
        f.create_dataset('vp',data=vp)
        f.create_dataset('snaps',data=u,compression='lzf')
        f.create_dataset('loc_x',data=np.int32(35+i))
# Optional fake six-case evaluation fixture. Again not a physical benchmark.
for name in ['Overthrust','SEGEAGE','Marmousi']:
    for shot in [1,2]:
        folder=root.parent/'test_cases'/name; folder.mkdir(parents=True,exist_ok=True)
        savemat(folder/f'shot{shot}.mat',{'vp':vp,'snaps':u})
print(root)
