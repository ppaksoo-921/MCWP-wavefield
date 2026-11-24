"""Read-only environment/data preflight. No installation and no GPU allocation."""
import argparse
import platform
from pathlib import Path
import h5py
import numpy as np
import scipy
import torch
from model import MediumConditionedWavePropagator

p = argparse.ArgumentParser()
p.add_argument('--data-dir', default=str(Path.home()/'Wage_surrogate/GenWP-pub/dataset/phaseA_outputs/h5'))
args = p.parse_args()
print('Python:', platform.python_version())
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available())
print('NumPy:', np.__version__, 'SciPy:', scipy.__version__, 'h5py:', h5py.__version__)
print('Model parameters:', sum(p.numel() for p in MediumConditionedWavePropagator().parameters()))
files = sorted(Path(args.data_dir).expanduser().rglob('*.h5'))
print('HDF5 trajectories:', len(files))
if not files:
    raise FileNotFoundError('No HDF5 files found.')
with h5py.File(files[0], 'r') as f:
    print('Example:', files[0])
    for name in ('snaps', 'vp', 'loc_x'):
        print(name, f[name].shape, f[name].dtype)
    assert f['snaps'].shape == (101,128,128)
    assert f['vp'].shape == (128,128)
print('PREFLIGHT PASS. This does not yet test CUDA training or all files.')
