"""Existing FD10 update order; no solver or numerical-method optimization."""
import importlib.util
from pathlib import Path
import sys


def load_fd(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('_final_mcwp_fd10', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def step(s, op, source, it):
    op.derx_backward_(s.Vx, s.dVx)
    op.derz_backward_(s.Vz, s.dVz)
    s.acoustic_iso_stress_torch(it, source)
    s.acoustic_iso_stress_plus_torch()
    op.derx_forward_(s.pressure, s.dpressure_Vx)
    op.derz_forward_(s.pressure, s.dpressure_Vz)
    s.acoustic_velocity_torch()
    s.sigmaxx0.copy_(s.sigmaxx)
    s.sigmazz0.copy_(s.sigmazz)
    s.Vx0.copy_(s.Vx)
    s.Vz0.copy_(s.Vz)


def generate(module, vp, loc, device, protocol):
    import numpy as np
    import torch
    c = protocol
    s = module.acoustic_iso_torch(
        dx=c['dx_m'], dz=c['dz_m'], dt=c['dt_s'], nt=c['nt'],
        loc_x_list=[int(loc)], freq=c['frequency_hz'], n_pml=c['pml_cells'],
        vel=vp, rho=np.ones_like(vp, dtype=np.float32), device=str(device), dtype=torch.float32)
    s.set_method('FD10')
    s.extend_wave_array()
    s.build_boundary()
    source = module.set_wavelet_torch(freq=c['frequency_hz'], dt=c['dt_s'], nt=c['nt'],
                                     device=str(device), dtype=torch.float32)
    op = module.FD10StaggeredConv2d(dx=c['dx_m'], dz=c['dz_m'], ext_cells=s.ext_cells,
                                   device=str(device), dtype=torch.float32).to(device)
    s.set_source_loc_torch(int(loc))
    s.reset_wavefields()
    stride = c['snapshot_stride']
    out = torch.empty(((c['nt']-1)//stride+1, *vp.shape), dtype=torch.float32, device=device)
    n = 0
    for it in range(c['nt']):
        step(s, op, source, it)
        if it % stride == 0:
            out[n].copy_(s.extract_matrix_torch())
            n += 1
    if n != len(out):
        raise ValueError('FD output-count mismatch.')
    return out
