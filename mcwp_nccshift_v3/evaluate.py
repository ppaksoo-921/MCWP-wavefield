"""Same 6 official cases for MCWP checkpoints and author/retrained GenWP weights.

Default GenWP noise is drawn ONCE per case, matching executable sample.py.
Reports both MAE(all 101 frames, official-code convention) and predicted-only
MAE(100 frames). No test labels enter recursive predictions except initial u[0].
"""
import argparse
import csv
import importlib
import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch

from common import trusted_load, hash_file, write_json
from model import ModelConfig, MediumConditionedWavePropagator

CASES = [(name, shot) for name in ('Overthrust', 'SEGEAGE', 'Marmousi') for shot in (1, 2)]
FIELDS = ['kind', 'label', 'steps', 'checkpoint', 'case', 'shot', 'seed', 'status',
          'mae_all101', 'mae_pred100', 'relative_l2_pred100', 'snr_db_pred100',
          'final_frame_mae', 'runtime_sec', 'parameters']
FRAME_FIELDS = ['kind', 'label', 'steps', 'case', 'shot', 'seed', 'frame', 'nominal_time_s', 'mae', 'relative_l2']


def load_case(path):
    d = sio.loadmat(path)
    vp = np.asarray(d['vp'], dtype=np.float32)
    snaps = np.asarray(d['snaps'], dtype=np.float32)
    original_nt = snaps.shape[0]
    if snaps.shape == (501, 128, 128):
        snaps = snaps[::5]
        stride = 5
    elif snaps.shape == (101, 128, 128):
        stride = 1
    else:
        raise ValueError(f'{path}: unexpected snaps shape {snaps.shape}; refusing to guess temporal stride.')
    if vp.shape != (128, 128) or not np.isfinite(snaps).all() or not np.isfinite(vp).all():
        raise ValueError(f'{path}: invalid velocity/wavefields.')
    return vp, snaps, {'original_frames': original_nt, 'applied_stride': stride}


def import_genwp(repo):
    folder = str((repo/'genwp').resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    # GenWP unfortunately calls its package 'code', shadowing Python's stdlib.
    existing = sys.modules.get('code')
    if existing is not None and not hasattr(existing, '__path__'):
        del sys.modules['code']
    module = importlib.import_module('code.script_util')
    if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError('A different code.script_util package was imported; run evaluate.py in a fresh Python process.')
    return module


def load_genwp(path, repo, device):
    util = import_genwp(repo)
    settings = util.model_and_diffusion_defaults()
    model, _ = util.create_model_and_diffusion(**settings, snap_steps=5)
    state = trusted_load(path)
    if isinstance(state, dict) and 'ema_params' in state and 'model' in state:
        raise ValueError('Pass GenWP EMA weights, not the full resume bundle. Extract EMA first.')
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), settings.get('out_channels', 1)


@torch.no_grad()
def rollout(model, kind, vp_np, ref_np, device, seed, noise_mode='fixed', noise_channels=1):
    vp = torch.as_tensor(vp_np.copy(), device=device)[None, None]
    truth = torch.as_tensor(ref_np.copy(), device=device)[None]
    prediction = torch.zeros_like(truth)
    prediction[:, 0] = truth[:, 0]
    history = torch.zeros(1, 5, 128, 128, device=device)
    history[:, 0] = truth[:, 0]
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, noise_channels, 128, 128, device=device, generator=generator) if kind == 'genwp' else None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    if kind == 'mcwp':
        medium = model.encode_medium(vp)
    else:
        normalized_vp = 2*(vp-1400)/3600-1
        diffusion_time = torch.full((1,), 0.999, device=device)
    status = 'PASS'
    for n in range(1, 101):
        index = torch.tensor([n], dtype=torch.long, device=device)
        if kind == 'mcwp':
            nxt = model.predict_next(history, index, medium)
        else:
            if noise_mode == 'fresh':
                z = torch.randn(1, noise_channels, 128, 128, device=device, generator=generator)
            nxt = model(z, torch.cat((history, normalized_vp), dim=1), index.float()*0.001, diffusion_time)[:, :1]
        if not torch.isfinite(nxt).all().item():
            prediction[:, n:] = float('nan')
            status = f'NONFINITE_AT_FRAME_{n}'
            break
        prediction[:, n:n+1] = nxt
        history = torch.cat((nxt, history[:, :-1]), dim=1)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return prediction[0].cpu().numpy(), time.perf_counter()-started, status


def metrics(pred, ref):
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    if not np.isfinite(a).all():
        return {k: float('nan') for k in ('mae_all101', 'mae_pred100', 'relative_l2_pred100', 'snr_db_pred100', 'final_frame_mae')}
    err = a-b
    num = np.sum(err[1:]**2)
    den = np.sum(b[1:]**2)
    return {'mae_all101': float(np.abs(err).mean()),
            'mae_pred100': float(np.abs(err[1:]).mean()),
            'relative_l2_pred100': float(np.sqrt(num/max(den, 1e-30))),
            'snr_db_pred100': float(10*np.log10(max(den, 1e-30)/max(num, 1e-30))),
            'final_frame_mae': float(np.abs(err[-1]).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', action='append', default=[], help='MCWP EMA or full checkpoint. Repeatable.')
    p.add_argument('--checkpoint-dir', action='append', default=[], help='Directory of MCWP ema_*.pt. Repeatable.')
    p.add_argument('--include-official', action='store_true')
    p.add_argument('--official-path', default='')
    p.add_argument('--genwp-checkpoint', action='append', nargs=2, metavar=('LABEL', 'PATH'), default=[])
    p.add_argument('--repo', default=str(Path.home()/'Wage_surrogate/GenWP-pub'))
    p.add_argument('--test-root', default='')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seeds', default='0')
    p.add_argument('--noise-mode', choices=['fixed', 'fresh'], default='fixed')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    device = torch.device('cpu' if args.cpu else 'cuda:0')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    repo = Path(args.repo).expanduser().resolve()
    root = Path(args.test_root).expanduser().resolve() if args.test_root else repo/'dataset/test'
    jobs = []
    paths = [Path(x).expanduser().resolve() for x in args.checkpoint]
    for folder in args.checkpoint_dir:
        found = sorted(Path(folder).expanduser().glob('ema_*.pt'))
        if not found:
            raise FileNotFoundError('No MCWP ema_*.pt in ' + folder)
        paths += [x.resolve() for x in found]
    for path in dict.fromkeys(paths):
        jobs.append(('mcwp', '', path))
    for label, path in args.genwp_checkpoint:
        jobs.append(('genwp', label, Path(path).expanduser().resolve()))
    if args.include_official:
        path = Path(args.official_path).expanduser().resolve() if args.official_path else repo/'genwp/checkpoints/trained_model.pt'
        jobs.append(('genwp', 'authors_final', path))
    if not jobs:
        raise ValueError('Select checkpoints and/or --include-official.')
    for _, _, path in jobs:
        if not path.is_file():
            raise FileNotFoundError(path)
    case_data = {case: load_case(root/case[0]/f'shot{case[1]}.mat') for case in CASES}
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A new output directory prevents mixing experiments/protocols by accident.
    if (out/'metrics.csv').exists():
        raise FileExistsError('metrics.csv already exists. Use a NEW --out-dir to preserve prior results.')
    write_json({'noise_mode': args.noise_mode, 'seeds': args.seeds, 'torch': torch.__version__,
                'test_root': str(root), 'source': 'official executable sample.py convention; z fixed per case by default',
                'mae_all101': 'includes copied initial condition; matches public sample.py accs',
                'mae_pred100': 'forecast frames 1..100 only',
                'cases': {f'{k[0]}_shot{k[1]}': v[2] for k, v in case_data.items()},
                'checkpoint_sha256': {str(path): hash_file(path) for _, _, path in jobs}}, out/'protocol.json')
    seeds = [int(x) for x in args.seeds.split(',')]
    with open(out/'metrics.csv', 'w', newline='') as mf, open(out/'per_frame.csv', 'w', newline='') as ff:
        writer, fw = csv.DictWriter(mf, fieldnames=FIELDS), csv.DictWriter(ff, fieldnames=FRAME_FIELDS)
        writer.writeheader(); fw.writeheader()
        for kind, label, path in jobs:
            if kind == 'mcwp':
                checkpoint = trusted_load(path)
                if checkpoint.get('kind') not in ('mcwp_full', 'mcwp_ema'):
                    raise ValueError('Not an MCWP checkpoint: ' + str(path))
                model = MediumConditionedWavePropagator(ModelConfig(**checkpoint['model_config']))
                model.load_state_dict(checkpoint['ema'] if checkpoint['kind'] == 'mcwp_full' else checkpoint['model'])
                model = model.to(device).eval()
                steps = checkpoint['completed_steps']
                label = f"{checkpoint['variant']}_{steps:06d}"
                channels = 1
                del checkpoint
            else:
                model, channels = load_genwp(path, repo, device)
                steps = ''  # Do not infer historical training budget from a filename.
            params = sum(x.numel() for x in model.parameters())
            print(f'\nEvaluating {label}: {path}', flush=True)
            for seed in seeds:
                for case_idx, (case, shot) in enumerate(CASES):
                    vp, ref, _ = case_data[(case, shot)]
                    # Same per-case noise for every GenWP checkpoint.
                    case_seed = seed + shot + case_idx*100
                    pred, seconds, status = rollout(model, kind, vp, ref, device, case_seed, args.noise_mode, channels)
                    row = dict(kind=kind, label=label, steps=steps, checkpoint=str(path), case=case,
                               shot=shot, seed=seed, status=status, runtime_sec=seconds, parameters=params)
                    row.update(metrics(pred, ref))
                    writer.writerow(row); mf.flush()
                    for n in range(101):
                        err = pred[n].astype(np.float64)-ref[n]
                        fw.writerow(dict(kind=kind, label=label, steps=steps, case=case, shot=shot,
                                         seed=seed, frame=n, nominal_time_s=0.01*n,
                                         mae=float(np.abs(err).mean()),
                                         relative_l2=float(np.sqrt(np.sum(err**2)/max(float(np.sum(ref[n].astype(np.float64)**2)), 1e-30)))))
                    ff.flush()
                    if args.save_predictions:
                        filename = f'{kind}_{label}_{case}_shot{shot}_seed{seed}.npz'
                        np.savez_compressed(out/filename, predict=pred, vp=vp, accs=row['mae_all101'])
                    print(f"{case} shot{shot} | {status} | MAE(all101)={row['mae_all101']:.6g} | "
                          f"MAE(pred100)={row['mae_pred100']:.6g} | SNR(pred100)={row['snr_db_pred100']:.3f} dB", flush=True)
            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    print('Saved: ' + str(out/'metrics.csv'), flush=True)


if __name__ == '__main__':
    main()
