#!/usr/bin/env python3
"""Final MCWP: native versus cached inference, paired with FD10 on paper cases."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
METHODS = ('FD10', 'MCWP_standard', 'MCWP_cached')
GROUPS = {'ID': 4, 'Marmousi_GenWP': 2, 'Marmousi_source': 15,
          'RedSea': 20, 'Kimberlina': 10, 'Pluto': 15}
FD = dict(dx_m=10.0, dz_m=10.0, dt_s=0.001, nt=1001, frequency_hz=15.0,
          pml_cells=50, snapshot_stride=10)
MAIN_SHA = 'fd54d4c20534d0b5960058fa9f9441308eaaa0c46f2a31e6d8a4e1901141d15e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def read_case(path):
    path = Path(path)
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            vp = np.asarray(f['vp'], np.float32).copy()
            ref = np.asarray(f['snaps'], np.float32).copy()
            loc = int(np.asarray(f['loc_x']).squeeze().item())
    elif path.suffix == '.mat':
        from scipy.io import loadmat
        f = loadmat(path)
        vp, ref = np.asarray(f['vp'], np.float32), np.asarray(f['snaps'], np.float32)
        loc = int(np.asarray(f['loc_x']).squeeze().item())
    else:
        raise ValueError('Expected the existing MAT/NPZ case: ' + str(path))
    if ref.shape == (501, 128, 128):
        ref = ref[::5].copy()
    if vp.shape != (128, 128) or ref.shape != (101, 128, 128):
        raise ValueError('Expected 128x128 velocity and 101 pressure fields: ' + str(path))
    if not np.isfinite(vp).all() or not np.isfinite(ref).all() or not np.any(ref[1:]):
        raise ValueError('Invalid or zero input: ' + str(path))
    if not 0 <= loc < 128:
        raise ValueError('Invalid source index: ' + str(path))
    return vp, ref, loc


def settings(args):
    c = read_json(args.config or HERE / 'config.json')
    root = Path(args.study_root or c['study_root']).expanduser().resolve()
    c['study_root'] = str(root)
    for key in ('checkpoint', 'fd_source', 'marmousi_run_dir'):
        value = getattr(args, key, None) or c[key]
        p = Path(value).expanduser()
        c[key] = str((root / p).resolve() if not p.is_absolute() else p.resolve())
    c['device'] = args.device or c['device']
    for k in ('warmup', 'repeats'):
        if getattr(args, k) is not None:
            c[k] = getattr(args, k)
        if c[k] < 1:
            raise ValueError(k + ' must be at least 1.')
    return c


def resolve_cases(c):
    catalog = read_json(HERE / 'paper_cases_final66.json')
    root = Path(c['study_root'])
    cases = []
    for item in catalog['cases']:
        if item['group'] == 'Marmousi_source':
            candidates = [Path(c['marmousi_run_dir']) / item['marmousi_relative']]
        else:
            candidates = [root / 'GenWP-pub' / item['repo_relative']]
            if item['group'] == 'Kimberlina':
                candidates.append(root / 'GenWP-pub/dataset/kimberlina_ood_year0/outputs' / (item['case'] + '.npz'))
            extension = '.mat' if item['group'] in ('ID', 'Marmousi_GenWP') else '.npz'
            candidates.append(root / '[dt]_variation/results/mcwp_dt03_all66_v1/case_links' /
                              item['group'] / (item['case'] + extension))
        override = c.get('case_overrides', {}).get(item['case'])
        if override:
            p = Path(override).expanduser()
            candidates.insert(0, (root / p).resolve() if not p.is_absolute() else p.resolve())
        selected = None
        tried = []
        for p in dict.fromkeys(candidates):
            if p.is_file():
                actual = sha(p)
                tried.append(str(p) + ' sha256=' + actual)
                if actual == item['expected_sha256']:
                    selected = p
                    break
            else:
                tried.append(str(p) + ' [missing]')
        if selected is None:
            raise FileNotFoundError('No exact final-paper input for ' + item['case'] +
                                    '\n' + '\n'.join(tried) +
                                    '\nUse config.json case_overrides for another location of the SAME file.')
        _, _, loc = read_case(selected)
        cases.append(dict(item, path=str(selected.resolve()), loc_x=loc))
    counts = dict(Counter(x['group'] for x in cases))
    if counts != GROUPS or len({x['case'] for x in cases}) != 66:
        raise ValueError('Expected the final six groups / 66 unique cases.')
    return cases


def plan(c, quiet=False):
    p = Path(c['checkpoint'])
    if not p.is_file():
        raise FileNotFoundError('Final MCWP checkpoint not found: ' + str(p))
    if sha(p) != MAIN_SHA:
        raise ValueError('Checkpoint differs from the final MCWP used in the paper. No fallback is used.')
    if not Path(c['fd_source']).is_file():
        raise FileNotFoundError('Existing FD10 source missing: ' + c['fd_source'])
    cases = resolve_cases(c)
    result = dict(checkpoint=c['checkpoint'], checkpoint_sha256=MAIN_SHA,
                  fd_source=c['fd_source'], fd_source_sha256=sha(c['fd_source']),
                  group_counts=GROUPS, cases=cases, protocol=FD,
                  native_cache='Native evaluator already computes the medium pyramid once.',
                  extra_cache='Medium modulation/gating and all target-time projections, once per trajectory.',
                  timing='FP32 eager; batch 1; all per-trajectory cache preparation included; no CUDA Graph.',
                  warmup=c['warmup'], repeats=c['repeats'])
    if not quiet:
        print('PLAN PASS: final MCWP checkpoint and all 66 final-paper input files verified. No GPU used.')
        for group, n in GROUPS.items():
            print(' ', group + ':', n)
        print('Checkpoint:', c['checkpoint'])
        print('Marmousi source:', c['marmousi_run_dir'])
    return result


def compare(a, b, label, relative_tol=1e-4, peak_tol=1e-4):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Shape/nonfinite parity failure: ' + label)
    d = b - a
    peak = max(float(np.abs(a).max()), 1e-30)
    norm = max(float(np.linalg.norm(a.ravel())), 1e-30)
    result = dict(check=label, mean_abs_difference=float(np.abs(d).mean()),
                  max_abs_difference=float(np.abs(d).max()),
                  relative_l2_difference=float(np.linalg.norm(d.ravel()) / norm),
                  max_difference_over_reference_peak=float(np.abs(d).max() / peak),
                  relative_l2_tolerance=relative_tol, peak_relative_tolerance=peak_tol)
    result['passed'] = bool(result['relative_l2_difference'] <= relative_tol and
                            result['max_difference_over_reference_peak'] <= peak_tol)
    return result


def metrics(pred, ref):
    # Frame zero is an input, so all accuracy and prediction parity use frames 1..100.
    p, r = np.asarray(pred[1:], np.float64), np.asarray(ref[1:], np.float64)
    error = p - r
    return dict(mae=float(np.abs(error).mean()), mse=float(np.square(error).mean()),
                relative_l2=float(np.sqrt(np.square(error).sum() / np.square(r).sum())))


def load_model(checkpoint, device):
    import torch
    from reference_model import ModelConfig, MediumConditionedWavePropagator
    kw = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kw['weights_only'] = False
    payload = torch.load(checkpoint, **kw)
    if payload.get('kind') != 'mcwp_ema' or payload.get('completed_steps', 4000) != 4000:
        raise ValueError('Expected the final MCWP EMA4k evaluation checkpoint.')
    cfg = ModelConfig(**payload['model_config'])
    if cfg.to_dict() != ModelConfig().to_dict():
        raise ValueError('Final-model architecture/normalization differs from the paper.')
    model = MediumConditionedWavePropagator(cfg)
    model.load_state_dict(payload['model'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 6307702:
        raise ValueError('Final-model parameter count mismatch.')
    return model.to(device).eval()


def timed_call(fn, device):
    import torch
    # Event initialization, synchronization and CPU output transfer are outside the primary interval.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(); end.record(); torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    stream = start.elapsed_time(end) / 1000.0
    t1 = time.perf_counter()
    host = result.detach().cpu().numpy().copy()
    transfer = time.perf_counter() - t1
    if host.shape != (101, 128, 128) or not np.isfinite(host).all():
        raise FloatingPointError('Invalid timed output.')
    return dict(generation_wall_s=wall, cuda_stream_interval_s=stream,
                cpu_output_s=transfer, generation_plus_cpu_output_s=wall+transfer)


def summarize(raw):
    per_case, by_group, overall = [], [], []
    fields = ('generation_wall_s', 'cuda_stream_interval_s', 'generation_plus_cpu_output_s')
    for case in dict.fromkeys(r['case'] for r in raw):
        for method in METHODS:
            rr = [r for r in raw if r['case'] == case and r['method'] == method]
            if not rr:
                raise ValueError('Incomplete paired timings: ' + case)
            row = dict(case=case, group=rr[0]['group'], method=method, repeats=len(rr))
            for key in fields:
                row[key] = statistics.median(r[key] for r in rr)
            per_case.append(row)
    groups = list(dict.fromkeys(r['group'] for r in raw))
    for group in groups:
        for method in METHODS:
            rr = [r for r in per_case if r['group'] == group and r['method'] == method]
            row = dict(group=group, method=method, cases=len(rr))
            for key in fields:
                row[key] = statistics.mean(r[key] for r in rr)
            by_group.append(row)
    for weighting in ('equal_group', 'equal_case'):
        source = by_group if weighting == 'equal_group' else per_case
        for method in METHODS:
            rr = [r for r in source if r['method'] == method]
            row = dict(weighting=weighting, method=method,
                       cases=sum(r['method'] == method for r in per_case), groups=len(groups))
            for key in fields:
                row[key] = statistics.mean(r[key] for r in rr)
            fd = next(r for r in overall if r['weighting'] == weighting and r['method'] == 'FD10') if method != 'FD10' else row
            row['fd_over_model_speedup'] = fd['generation_wall_s'] / row['generation_wall_s']
            overall.append(row)
    for row in per_case:
        fd = next(r for r in per_case if r['case'] == row['case'] and r['method'] == 'FD10')
        row['fd_over_model_speedup'] = fd['generation_wall_s'] / row['generation_wall_s']
    return per_case, by_group, overall


def package_report(out):
    target = out.parent / (out.name + '_report.zip')
    with ZipFile(target, 'w', ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out))
        for p in [HERE/'run_cache.py', HERE/'cache_ops.py', HERE/'fd_adapter.py',
                  HERE/'reference_model.py', HERE/'paper_cases_final66.json', HERE/'config.json']:
            z.write(p, 'benchmark_code/' + p.name)
    print('Report ZIP (no weights or wavefield arrays):', target, flush=True)
    return target


def save_summary(out):
    with (out/'timings_raw.csv').open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    for row in raw:
        for k in ('generation_wall_s', 'cuda_stream_interval_s', 'generation_plus_cpu_output_s'):
            row[k] = float(row[k])
    per_case, groups, overall = summarize(raw)
    write_csv(out/'summary_per_case.csv', per_case)
    write_csv(out/'summary_by_group.csv', groups)
    write_csv(out/'summary_overall.csv', overall)
    with (out/'parity_checks.csv').open(encoding='utf-8-sig', newline='') as f:
        checks = [r for r in csv.DictReader(f) if r['check'] == 'Cached versus standard MCWP']
    macro = {r['method']:r for r in overall if r['weighting'] == 'equal_group'}
    cached = [r for r in per_case if r['method'] == 'MCWP_cached']
    headline = dict(cases=len(cached), groups=len(groups)//len(METHODS),
        aggregation='Equal group weighting of per-case median wall times.',
        mean_time_reduction_vs_fd_percent=100*(1-macro['MCWP_cached']['generation_wall_s']/macro['FD10']['generation_wall_s']),
        mean_time_reduction_vs_standard_percent=100*(1-macro['MCWP_cached']['generation_wall_s']/macro['MCWP_standard']['generation_wall_s']),
        cached_faster_than_fd_cases=sum(r['fd_over_model_speedup'] > 1 for r in cached),
        max_case_mean_abs_prediction_difference=max(float(r['mean_abs_difference']) for r in checks),
        max_absolute_prediction_difference=max(float(r['max_abs_difference']) for r in checks),
        max_relative_l2_prediction_difference=max(float(r['relative_l2_difference']) for r in checks))
    write_json(out/'headline.json', headline)
    for row in [r for r in overall if r['weighting'] == 'equal_group']:
        print('%s: %.3f ms; FD/model = %.3f' %
              (row['method'], row['generation_wall_s']*1000, row['fd_over_model_speedup']), flush=True)
    return package_report(out)


def run(c, smoke=False):
    inventory = plan(c, quiet=True)
    import torch
    from cache_ops import predict
    from fd_adapter import generate, load_fd
    device = torch.device(c['device'])
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Run the benchmark in the existing CUDA environment.')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') not in ('', '0'):
        raise RuntimeError('Unset CUDA_LAUNCH_BLOCKING for a normal runtime measurement.')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(c['checkpoint'], device)
    fd_module = load_fd(c['fd_source'])
    selected = inventory['cases']
    if smoke:
        selected = [next(x for x in selected if x['group'] == g) for g in GROUPS]
    warmup, repeats = (1, 1) if smoke else (c['warmup'], c['repeats'])
    out = HERE/'results'/datetime.now().strftime(('%Y%m%d_%H%M%S_%f_') + ('smoke' if smoke else 'full'))
    out.mkdir(parents=True, exist_ok=False)
    protocol = dict(status='running', smoke=smoke, cases=len(selected), effective_config=c,
        inventory=inventory, selected_cases=selected, methods=list(METHODS), warmup=warmup, repeats=repeats,
        environment=dict(python=sys.version, os=platform.platform(), torch=torch.__version__,
            pytorch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(device), device=str(device),
            torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            fp32=True, tf32=False, cuda_graph=False, cudnn_benchmark=False),
        code_sha256={p:sha(HERE/p) for p in ('run_cache.py','cache_ops.py','fd_adapter.py','reference_model.py')},
        timing_definition=dict(primary='Synchronized wall time, CPU-resident Vp/initial field to GPU-resident 101-field trajectory.',
            fd_included='Material/PML/wavelet construction, allocation, reset and FD updates 0..1000; store every tenth field.',
            mcwp_included='Input transfer, history/output allocation, medium encoding, all extra caching once, 100 recursive updates.',
            excluded='Disk I/O, checkpoint loading, warm-up, parity/accuracy checks, final CPU transfer.',
            initial_condition='MCWP is conditioned on supplied FD frame 0 and four zero older fields.',
            cross_shot_amortization=False, accuracy_frames='1..100; frame 0 excluded',
            secondary='Primary plus a final transfer and owned CPU NumPy copy of all 101 fields.',
            stream_interval='Auxiliary event interval; includes GPU idle gaps and is not summed kernel execution time.',
            aggregation='Median across repeats per case, then group means, then equal weights for six groups.'))
    write_json(out/'protocol.json', protocol)
    raw, parity, accuracy = [], [], []
    print('Results:', out, flush=True)
    try:
        with torch.no_grad():
            for ci, item in enumerate(selected, 1):
                print('CASE %d/%d %s' % (ci, len(selected), item['case']), flush=True)
                if sha(item['path']) != item['expected_sha256']:
                    raise ValueError('Case file changed after plan: ' + item['path'])
                vp, ref, loc = read_case(item['path'])
                functions = {
                    'FD10': lambda: generate(fd_module, vp, loc, device, FD),
                    'MCWP_standard': lambda: predict(model, vp, ref, device, False),
                    'MCWP_cached': lambda: predict(model, vp, ref, device, True)}
                actual = {name: fn().detach().cpu().numpy().copy() for name, fn in functions.items()}
                checks = [compare(ref, actual['FD10'], 'FD10 versus saved FD reference'),
                          compare(actual['MCWP_standard'][1:], actual['MCWP_cached'][1:], 'Cached versus standard MCWP')]
                for check in checks:
                    parity.append(dict(case=item['case'], group=item['group'], **check))
                write_csv(out/'parity_checks.csv', parity)
                if not all(r['passed'] for r in checks):
                    raise ValueError('Numerical parity failed. See parity_checks.csv; no speed claim should be made.')
                for method, prediction in actual.items():
                    accuracy.append(dict(case=item['case'], group=item['group'], method=method,
                                         **metrics(prediction, ref)))
                expected = item.get('expected_mcwp_mae')
                if expected is not None:
                    measured = accuracy[-2]['mae']
                    if not np.isclose(measured, expected, rtol=0.002, atol=1e-7):
                        raise ValueError('Native MCWP accuracy differs from the final paper for ' + item['case'] +
                                         ': got %.9g, expected %.9g' % (measured, expected))
                write_csv(out/'accuracy_per_case.csv', accuracy)
                del actual
                for _ in range(warmup):
                    for method in METHODS:
                        result = functions[method]()
                        torch.cuda.synchronize(device)
                        del result
                for repeat in range(repeats):
                    order = list(METHODS)
                    random.Random(20261024 + ci*100 + repeat).shuffle(order)
                    for rank, method in enumerate(order, 1):
                        row = dict(case=item['case'], group=item['group'], method=method,
                                   repeat=repeat+1, order=rank, **timed_call(functions[method], device))
                        raw.append(row)
                        print('  %s %d/%d: %.3f ms' %
                              (method, repeat+1, repeats, row['generation_wall_s']*1000), flush=True)
                    write_csv(out/'timings_raw.csv', raw)
                del functions
        if sha(c['checkpoint']) != MAIN_SHA:
            raise ValueError('Final checkpoint changed during the benchmark.')
        protocol.update(status='complete', measurements=len(raw))
        write_json(out/'protocol.json', protocol)
        write_json(HERE/'results'/('latest_smoke.json' if smoke else 'latest_full.json'), {'run_dir':str(out)})
        save_summary(out)
    except BaseException as exc:
        protocol.update(status='incomplete', error=str(exc), measurements=len(raw))
        write_json(out/'protocol.json', protocol)
        write_csv(out/'timings_raw.csv', raw)
        write_csv(out/'accuracy_per_case.csv', accuracy)
        write_csv(out/'parity_checks.csv', parity)
        package_report(out)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'smoke', 'all', 'report'))
    p.add_argument('--config', type=Path)
    p.add_argument('--study-root')
    p.add_argument('--checkpoint')
    p.add_argument('--fd-source')
    p.add_argument('--marmousi-run-dir')
    p.add_argument('--device')
    p.add_argument('--warmup', type=int)
    p.add_argument('--repeats', type=int)
    p.add_argument('--run-dir', type=Path)
    a = p.parse_args()
    if a.stage == 'report':
        out = a.run_dir.resolve() if a.run_dir else Path(read_json(HERE/'results/latest_full.json')['run_dir'])
        if not out.is_relative_to(HERE/'results'):
            raise ValueError('Select a run inside this package results folder.')
        if read_json(out/'protocol.json')['status'] != 'complete':
            raise ValueError('Incomplete benchmark: inspect its failure report, then rerun.')
        save_summary(out)
    else:
        c = settings(a)
        if a.stage == 'plan':
            plan(c)
        else:
            run(c, smoke=a.stage == 'smoke')


if __name__ == '__main__':
    main()
