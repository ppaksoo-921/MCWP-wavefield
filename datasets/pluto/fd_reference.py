"""FD10 snapshot loop from the supplied common research benchmark.

The computational loop and source timing are retained. The installed GenWP
fd_torch.py supplies the numerical operators; it is not replaced by this file.
"""
import importlib
import sys
from pathlib import Path
import numpy as np
import torch

def import_fd(repo):
    folder = str((repo / 'dataset/data_gen').resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    module = importlib.import_module('fd_torch')
    if Path(module.__file__).resolve().parent != Path(folder):
        raise RuntimeError(f'Wrong fd_torch imported: {module.__file__}')
    return module.acoustic_iso_torch, module.FD10StaggeredConv2d, module.set_wavelet_torch

def build_fd(repo, vp_np, loc_x, device, args):
    acoustic_iso_torch, FD10StaggeredConv2d, set_wavelet_torch = import_fd(repo)
    rho = np.ones_like(vp_np, dtype=np.float32)
    solver = acoustic_iso_torch(
        dx=args.fd_dx, dz=args.fd_dz, dt=args.fd_dt, nt=args.fd_nt,
        loc_x_list=[int(loc_x)], freq=args.fd_freq, n_pml=args.fd_pml,
        vel=vp_np, rho=rho, device=str(device), dtype=torch.float32,
    )
    solver.set_method('FD10')
    solver.extend_wave_array()
    solver.build_boundary()
    src = set_wavelet_torch(freq=args.fd_freq, dt=args.fd_dt, nt=args.fd_nt,
        device=str(device), dtype=torch.float32)
    fdop = FD10StaggeredConv2d(dx=args.fd_dx, dz=args.fd_dz,
        ext_cells=solver.ext_cells, device=str(device), dtype=torch.float32).to(device)
    return solver, src, fdop

@torch.no_grad()
def fd_predict(repo, vp_np, loc_x, device, args):
    solver, src, fdop = build_fd(repo, vp_np, loc_x, device, args)
    solver.set_source_loc_torch(int(loc_x))
    solver.reset_wavefields()
    nout = (solver.nt - 1) // args.fd_downsample + 1
    storage = torch.empty((nout, solver.nz, solver.nx), dtype=solver.dtype, device=solver.device)
    out_idx = 0
    for it in range(solver.nt):
        fdop.derx_backward_(solver.Vx, solver.dVx)
        fdop.derz_backward_(solver.Vz, solver.dVz)
        solver.acoustic_iso_stress_torch(it, src)
        solver.acoustic_iso_stress_plus_torch()
        fdop.derx_forward_(solver.pressure, solver.dpressure_Vx)
        fdop.derz_forward_(solver.pressure, solver.dpressure_Vz)
        solver.acoustic_velocity_torch()
        solver.sigmaxx0.copy_(solver.sigmaxx)
        solver.sigmazz0.copy_(solver.sigmazz)
        solver.Vx0.copy_(solver.Vx)
        solver.Vz0.copy_(solver.Vz)
        if it % args.fd_downsample == 0:
            storage[out_idx].copy_(solver.extract_matrix_torch())
            out_idx += 1
    return storage
