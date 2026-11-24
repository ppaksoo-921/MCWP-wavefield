"""Evaluate MCWP / ablation models and GenWP on either:

1) the original 6 official cases (default), or
2) Red Sea OOD NPZ cases selected with --redsea-root.

For Red Sea OOD, files are discovered automatically using the naming pattern
RedSea_vel<id>_shot<id>.npz.

The evaluator preserves the original rollout protocol:
- 101 frames total = copied initial frame + 100 recursive predictions
- no test labels enter recursive prediction after frame 0
- GenWP noise is fixed once per case by default
- GenWP velocity normalization remains exactly 2*(vp-1400)/3600 - 1
  with no clipping, including for OOD velocities outside the training range.
"""

import argparse
import csv
import importlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from common import trusted_load, hash_file, write_json
from reference_model import ModelConfig, MediumConditionedWavePropagator
from model import build_model


OFFICIAL_CASES = [
    (name, shot)
    for name in ('Overthrust', 'SEGEAGE', 'Marmousi')
    for shot in (1, 2)
]

FIELDS = [
    'kind', 'label', 'steps', 'checkpoint', 'case', 'shot', 'seed', 'status',
    'mae_all101', 'mae_pred100', 'relative_l2_pred100', 'snr_db_pred100',
    'final_frame_mae', 'runtime_sec', 'parameters'
]

FRAME_FIELDS = [
    'kind', 'label', 'steps', 'case', 'shot', 'seed', 'frame',
    'nominal_time_s', 'mae', 'relative_l2'
]


REDSEA_PATTERN = re.compile(r'^RedSea_vel(\d+)_shot(\d+)\.npz$')


def load_case(path):
    """Load one .mat or .npz case and return vp, 101 wavefield frames, metadata."""
    path = Path(path)

    if path.suffix.lower() == '.mat':
        d = sio.loadmat(path)
        if 'vp' not in d or 'snaps' not in d:
            raise KeyError(f'{path}: expected keys vp and snaps.')
        vp = np.asarray(d['vp'], dtype=np.float32)
        snaps = np.asarray(d['snaps'], dtype=np.float32)
        loc_x = None
        source_format = 'mat'

    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as d:
            if 'vp' not in d.files or 'snaps' not in d.files:
                raise KeyError(f'{path}: expected keys vp and snaps; found {d.files}.')
            vp = np.asarray(d['vp'], dtype=np.float32)
            snaps = np.asarray(d['snaps'], dtype=np.float32)

            if 'loc_x' in d.files:
                raw_loc_x = np.asarray(d['loc_x'])
                if raw_loc_x.size == 1:
                    loc_x = int(raw_loc_x.reshape(-1)[0])
                else:
                    loc_x = raw_loc_x.tolist()
            else:
                loc_x = None
        source_format = 'npz'

    else:
        raise ValueError(f'{path}: unsupported file extension {path.suffix}.')

    original_nt = int(snaps.shape[0]) if snaps.ndim >= 1 else -1

    if snaps.shape == (501, 128, 128):
        snaps = snaps[::5]
        stride = 5
    elif snaps.shape == (101, 128, 128):
        stride = 1
    else:
        raise ValueError(
            f'{path}: unexpected snaps shape {snaps.shape}; '
            'expected (501,128,128) or (101,128,128).'
        )

    if vp.shape != (128, 128):
        raise ValueError(f'{path}: unexpected vp shape {vp.shape}; expected (128,128).')
    if not np.isfinite(snaps).all() or not np.isfinite(vp).all():
        raise ValueError(f'{path}: non-finite values found in velocity/wavefields.')

    meta = {
        'path': str(path.resolve()),
        'format': source_format,
        'original_frames': original_nt,
        'applied_stride': stride,
        'loc_x': loc_x,
        'vp_min': float(vp.min()),
        'vp_max': float(vp.max()),
        'snaps_min': float(snaps.min()),
        'snaps_max': float(snaps.max()),
    }
    return vp, snaps, meta


def discover_redsea_cases(root):
    """Discover RedSea_vel<id>_shot<id>.npz files and sort numerically."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    discovered = []
    for path in root.glob('RedSea_vel*_shot*.npz'):
        m = REDSEA_PATTERN.match(path.name)
        if m is None:
            continue
        vel_id = int(m.group(1))
        shot = int(m.group(2))
        discovered.append((vel_id, shot, path.resolve()))

    discovered.sort(key=lambda x: (x[0], x[1]))

    if not discovered:
        raise FileNotFoundError(
            f'No RedSea_vel<id>_shot<id>.npz files found in {root}'
        )

    seen = set()
    cases = []
    for vel_id, shot, path in discovered:
        key = (vel_id, shot)
        if key in seen:
            raise ValueError(f'Duplicate Red Sea case for vel{vel_id} shot{shot}.')
        seen.add(key)
        cases.append((f'RedSea_vel{vel_id}', shot, path))

    return cases


def build_case_list(args, repo):
    """Return list of (case_name, shot, path) and dataset label."""
    if args.redsea_root:
        cases = discover_redsea_cases(args.redsea_root)
        dataset_label = 'redsea_ood'
    else:
        root = (
            Path(args.test_root).expanduser().resolve()
            if args.test_root
            else repo / 'dataset' / 'test'
        )
        cases = [
            (case, shot, root / case / f'shot{shot}.mat')
            for case, shot in OFFICIAL_CASES
        ]
        dataset_label = 'official_6_cases'

    if args.limit_cases > 0:
        cases = cases[:args.limit_cases]

    if not cases:
        raise ValueError('No evaluation cases selected.')

    for _, _, path in cases:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    return cases, dataset_label


def import_genwp(repo):
    folder = str((repo / 'genwp').resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)

    # GenWP calls its package 'code', shadowing Python's stdlib module.
    existing = sys.modules.get('code')
    if existing is not None and not hasattr(existing, '__path__'):
        del sys.modules['code']

    module = importlib.import_module('code.script_util')
    if not Path(module.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError(
            'A different code.script_util package was imported; '
            'run evaluate.py in a fresh Python process.'
        )
    return module


def load_genwp(path, repo, device):
    util = import_genwp(repo)
    settings = util.model_and_diffusion_defaults()
    model, _ = util.create_model_and_diffusion(**settings, snap_steps=5)

    state = trusted_load(path)
    if isinstance(state, dict) and 'ema_params' in state and 'model' in state:
        raise ValueError(
            'Pass GenWP EMA weights, not the full resume bundle. Extract EMA first.'
        )
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), settings.get('out_channels', 1)


@torch.no_grad()
def rollout(
    model,
    kind,
    vp_np,
    ref_np,
    device,
    seed,
    noise_mode='fixed',
    noise_channels=1,
):
    vp = torch.as_tensor(vp_np.copy(), device=device)[None, None]
    truth = torch.as_tensor(ref_np.copy(), device=device)[None]

    prediction = torch.zeros_like(truth)
    prediction[:, 0] = truth[:, 0]

    history = torch.zeros(1, 5, 128, 128, device=device)
    history[:, 0] = truth[:, 0]

    generator = torch.Generator(device=device).manual_seed(seed)
    z = (
        torch.randn(
            1, noise_channels, 128, 128,
            device=device, generator=generator
        )
        if kind == 'genwp'
        else None
    )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    if kind == 'mcwp':
        medium = model.encode_medium(vp)
    else:
        # IMPORTANT: preserve the original GenWP normalization exactly.
        # No clipping is applied, even for OOD vp values outside [1400, 5000].
        normalized_vp = 2 * (vp - 1400) / 3600 - 1
        diffusion_time = torch.full((1,), 0.999, device=device)

    status = 'PASS'

    for n in range(1, 101):
        index = torch.tensor([n], dtype=torch.long, device=device)

        if kind == 'mcwp':
            nxt = model.predict_next(history, index, medium)
        else:
            if noise_mode == 'fresh':
                z = torch.randn(
                    1, noise_channels, 128, 128,
                    device=device, generator=generator
                )
            nxt = model(
                z,
                torch.cat((history, normalized_vp), dim=1),
                index.float() * 0.001,
                diffusion_time,
            )[:, :1]

        if not torch.isfinite(nxt).all().item():
            prediction[:, n:] = float('nan')
            status = f'NONFINITE_AT_FRAME_{n}'
            break

        prediction[:, n:n+1] = nxt
        history = torch.cat((nxt, history[:, :-1]), dim=1)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    return (
        prediction[0].cpu().numpy(),
        time.perf_counter() - started,
        status,
    )


def metrics(pred, ref):
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)

    if not np.isfinite(a).all():
        return {
            k: float('nan')
            for k in (
                'mae_all101',
                'mae_pred100',
                'relative_l2_pred100',
                'snr_db_pred100',
                'final_frame_mae',
            )
        }

    err = a - b
    num = np.sum(err[1:] ** 2)
    den = np.sum(b[1:] ** 2)

    return {
        'mae_all101': float(np.abs(err).mean()),
        'mae_pred100': float(np.abs(err[1:]).mean()),
        'relative_l2_pred100': float(np.sqrt(num / max(den, 1e-30))),
        'snr_db_pred100': float(
            10 * np.log10(max(den, 1e-30) / max(num, 1e-30))
        ),
        'final_frame_mae': float(np.abs(err[-1]).mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        '--checkpoint',
        action='append',
        default=[],
        help='Ablation or original MCWP EMA/full checkpoint. Repeatable.',
    )
    p.add_argument(
        '--checkpoint-dir',
        action='append',
        default=[],
        help='Directory of MCWP ema_*.pt. Repeatable.',
    )
    p.add_argument('--include-official', action='store_true')
    p.add_argument('--official-path', default='')
    p.add_argument(
        '--genwp-checkpoint',
        action='append',
        nargs=2,
        metavar=('LABEL', 'PATH'),
        default=[],
    )

    p.add_argument(
        '--repo',
        default=str(Path.home() / 'Wage_surrogate/GenWP-pub'),
    )
    p.add_argument(
        '--test-root',
        default='',
        help='Official .mat test root. Ignored when --redsea-root is used.',
    )
    p.add_argument(
        '--redsea-root',
        default='',
        help=(
            'Directory containing RedSea_vel<id>_shot<id>.npz. '
            'When set, evaluates Red Sea OOD instead of the official 6 cases.'
        ),
    )
    p.add_argument(
        '--limit-cases',
        type=int,
        default=0,
        help='Evaluate only the first N discovered cases. 0 = all cases.',
    )
    p.add_argument(
        '--nominal-dt',
        type=float,
        default=0.01,
        help='Nominal seconds between the 101 evaluated frames.',
    )

    p.add_argument('--out-dir', required=True)
    p.add_argument('--seeds', default='0')
    p.add_argument('--noise-mode', choices=['fixed', 'fresh'], default='fixed')
    p.add_argument('--save-predictions', action='store_true')
    p.add_argument('--cpu', action='store_true')
    p.add_argument(
        '--warmup-rollouts',
        type=int,
        default=1,
        help=(
            'Same full-rollout warmups for EVERY model; '
            '0 reproduces old evaluator timing.'
        ),
    )

    args = p.parse_args()

    if args.warmup_rollouts < 0:
        raise ValueError('warmup-rollouts must be nonnegative.')
    if args.limit_cases < 0:
        raise ValueError('limit-cases must be nonnegative.')
    if args.nominal_dt <= 0:
        raise ValueError('nominal-dt must be positive.')

    device = torch.device('cpu' if args.cpu else 'cuda:0')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    repo = Path(args.repo).expanduser().resolve()

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
        jobs.append(
            ('genwp', label, Path(path).expanduser().resolve())
        )

    if args.include_official:
        path = (
            Path(args.official_path).expanduser().resolve()
            if args.official_path
            else repo / 'genwp' / 'checkpoints' / 'trained_model.pt'
        )
        jobs.append(('genwp', 'authors_final', path))

    if not jobs:
        raise ValueError('Select checkpoints and/or --include-official.')

    for _, _, path in jobs:
        if not path.is_file():
            raise FileNotFoundError(path)

    cases, dataset_label = build_case_list(args, repo)

    case_data = {}
    for case, shot, path in cases:
        key = (case, shot)
        if key in case_data:
            raise ValueError(f'Duplicate case key: {key}')
        case_data[key] = load_case(path)

    print(f'\nDataset: {dataset_label}', flush=True)
    print(f'Number of evaluation cases: {len(cases)}', flush=True)
    for case, shot, path in cases:
        vp, ref, meta = case_data[(case, shot)]
        print(
            f'  {case} shot{shot}: {Path(path).name} | '
            f'vp=[{vp.min():.1f}, {vp.max():.1f}] | '
            f'snaps={ref.shape} | loc_x={meta["loc_x"]}',
            flush=True,
        )

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # A new output directory prevents mixing experiments/protocols by accident.
    if (out / 'metrics.csv').exists():
        raise FileExistsError(
            'metrics.csv already exists. Use a NEW --out-dir to preserve prior results.'
        )

    seeds = list(
        dict.fromkeys(
            int(x.strip())
            for x in args.seeds.split(',')
            if x.strip()
        )
    )
    if not seeds:
        raise ValueError('At least one evaluation seed is required.')

    protocol_cases = {}
    for case, shot, path in cases:
        protocol_cases[f'{case}_shot{shot}'] = case_data[(case, shot)][2]

    write_json(
        {
            'dataset': dataset_label,
            'noise_mode': args.noise_mode,
            'seeds': args.seeds,
            'torch': torch.__version__,
            'repo': str(repo),
            'test_root': str(Path(args.test_root).expanduser().resolve())
            if args.test_root
            else '',
            'redsea_root': str(Path(args.redsea_root).expanduser().resolve())
            if args.redsea_root
            else '',
            'source': (
                'official executable sample.py convention; '
                'z fixed per case by default'
            ),
            'mae_all101': (
                'includes copied initial condition; '
                'matches public sample.py accs'
            ),
            'mae_pred100': 'forecast frames 1..100 only',
            'nominal_dt_s': args.nominal_dt,
            'genwp_vp_normalization': '2*(vp-1400)/3600-1; no clipping',
            'cases': protocol_cases,
            'checkpoint_sha256': {
                str(path): hash_file(path)
                for _, _, path in jobs
            },
            'warmup_rollouts_per_model': args.warmup_rollouts,
            'timing_note': (
                'Same evaluator checks and device synchronization for all models. '
                'Single-case timing is diagnostic, not a repeated standalone '
                'runtime benchmark.'
            ),
        },
        out / 'protocol.json',
    )

    seen_labels = set()

    with (
        open(out / 'metrics.csv', 'w', newline='') as mf,
        open(out / 'per_frame.csv', 'w', newline='') as ff,
    ):
        writer = csv.DictWriter(mf, fieldnames=FIELDS)
        fw = csv.DictWriter(ff, fieldnames=FRAME_FIELDS)
        writer.writeheader()
        fw.writeheader()

        for kind, label, path in jobs:
            if kind == 'mcwp':
                checkpoint = trusted_load(path)
                checkpoint_kind = checkpoint.get('kind')
                config = ModelConfig(**checkpoint['model_config'])

                if checkpoint_kind in (
                    'mcwp_ablation_full',
                    'mcwp_ablation_ema',
                ):
                    model = build_model(config, checkpoint['variant'])
                    label_prefix = 'ablation_' + checkpoint['variant']

                elif checkpoint_kind in ('mcwp_full', 'mcwp_ema'):
                    model = MediumConditionedWavePropagator(config)
                    label_prefix = checkpoint['variant']

                else:
                    raise ValueError(
                        'Not a supported MCWP/ablation checkpoint: ' + str(path)
                    )

                is_full = checkpoint_kind.endswith('_full')
                model.load_state_dict(
                    checkpoint['ema'] if is_full else checkpoint['model'],
                    strict=True,
                )
                model = model.to(device).eval()
                steps = checkpoint['completed_steps']
                label = f'{label_prefix}_{steps:06d}'
                channels = 1
                del checkpoint

            else:
                model, channels = load_genwp(path, repo, device)
                steps = ''

            if label in seen_labels:
                raise ValueError(
                    'Duplicate model label in one evaluation. '
                    'Use separate output folders for different training seeds: '
                    + label
                )
            seen_labels.add(label)

            params = sum(x.numel() for x in model.parameters())
            print(f'\nEvaluating {label}: {path}', flush=True)

            warm_case, warm_shot, _ = cases[0]
            warm_vp, warm_ref, _ = case_data[(warm_case, warm_shot)]
            for warmup in range(args.warmup_rollouts):
                rollout(
                    model,
                    kind,
                    warm_vp,
                    warm_ref,
                    device,
                    seeds[0] + warm_shot,
                    args.noise_mode,
                    channels,
                )

            for seed in seeds:
                for case_idx, (case, shot, _) in enumerate(cases):
                    vp, ref, _ = case_data[(case, shot)]

                    # Same per-case noise for every GenWP checkpoint.
                    case_seed = seed + shot + case_idx * 100

                    pred, seconds, status = rollout(
                        model,
                        kind,
                        vp,
                        ref,
                        device,
                        case_seed,
                        args.noise_mode,
                        channels,
                    )

                    row = dict(
                        kind=kind,
                        label=label,
                        steps=steps,
                        checkpoint=str(path),
                        case=case,
                        shot=shot,
                        seed=seed,
                        status=status,
                        runtime_sec=seconds,
                        parameters=params,
                    )
                    row.update(metrics(pred, ref))
                    writer.writerow(row)
                    mf.flush()

                    for n in range(101):
                        if np.isfinite(pred[n]).all():
                            err = pred[n].astype(np.float64) - ref[n].astype(np.float64)
                            frame_mae = float(np.abs(err).mean())
                            frame_rel_l2 = float(
                                np.sqrt(
                                    np.sum(err ** 2)
                                    / max(
                                        float(
                                            np.sum(
                                                ref[n].astype(np.float64) ** 2
                                            )
                                        ),
                                        1e-30,
                                    )
                                )
                            )
                        else:
                            frame_mae = float('nan')
                            frame_rel_l2 = float('nan')

                        fw.writerow(
                            dict(
                                kind=kind,
                                label=label,
                                steps=steps,
                                case=case,
                                shot=shot,
                                seed=seed,
                                frame=n,
                                nominal_time_s=args.nominal_dt * n,
                                mae=frame_mae,
                                relative_l2=frame_rel_l2,
                            )
                        )
                    ff.flush()

                    if args.save_predictions:
                        filename = (
                            f'{kind}_{label}_{case}_shot{shot}_seed{seed}.npz'
                        )
                        np.savez_compressed(
                            out / filename,
                            predict=pred,
                            vp=vp,
                            accs=row['mae_all101'],
                        )

                    print(
                        f'{case} shot{shot} | {status} | '
                        f'MAE(all101)={row["mae_all101"]:.6g} | '
                        f'MAE(pred100)={row["mae_pred100"]:.6g} | '
                        f'SNR(pred100)={row["snr_db_pred100"]:.3f} dB',
                        flush=True,
                    )

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print('\nSaved: ' + str(out / 'metrics.csv'), flush=True)
    print('Saved: ' + str(out / 'per_frame.csv'), flush=True)
    print('Saved: ' + str(out / 'protocol.json'), flush=True)


if __name__ == '__main__':
    main()
