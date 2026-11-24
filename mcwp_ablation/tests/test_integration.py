"""Tiny CPU training/resume/DDP/evaluation tests on SOFTWARE fixtures, not physics validation."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import trusted_load
from model import REQUIRED


def equal_trees(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_trees(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_trees(v, w) for v, w in zip(a, b))
    return a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', default='')
    p.add_argument('--reference-root', default='')
    args = p.parse_args()
    tmp = None if args.work_dir else tempfile.TemporaryDirectory(prefix='mcwp_ablation_tests_')
    work = Path(args.work_dir or tmp.name).resolve(); work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    env.pop('MASTER_PORT', None)
    env.pop('CUDA_VISIBLE_DEVICES', None)
    result = []

    def execute(cmd, name, success=True, timeout=180):
        with open(work/(name+'.log'), 'w') as out:
            r = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        if (r.returncode == 0) != success:
            raise RuntimeError(f'{name} failed unexpectedly; log:\n' + (work/(name+'.log')).read_text())
        return r

    execute([sys.executable, str(ROOT/'tests/make_fixture.py'), str(work/'h5')], 'fixture')

    def train(variant, name, steps, resume='fresh', world=1, workers=0, entry=None, success=True):
        cmd = [sys.executable]
        if world > 1:
            cmd += ['-m', 'torch.distributed.run', '--rdzv_backend=c10d', '--rdzv_endpoint=localhost:0',
                    '--rdzv_id='+name, '--nnodes=1', '--nproc_per_node='+str(world)]
        cmd += [str(entry or ROOT/'train.py'), '--variant', variant, '--data-dir', str(work/'h5'),
                '--run-dir', str(work/name), '--cpu', '--width', '4', '--modes', '2',
                '--global-batch', '4', '--workers', str(workers), '--max-steps', str(steps),
                '--log-every', '1', '--save-every', '2', '--eval-every', '2', '--eval-at', '',
                '--preserve-full', '', '--resume', resume]
        execute(cmd, f'{name}_{steps}_{resume}', success=success)

    for variant in REQUIRED:
        train(variant, variant+'_continuous', 4)
        train(variant, variant+'_resumed', 2)
        train(variant, variant+'_resumed', 4, resume='auto')
        a = trusted_load(work/(variant+'_continuous/checkpoints/resume_000004.pt'))
        b = trusted_load(work/(variant+'_resumed/checkpoints/resume_000004.pt'))
        for part in ('model', 'ema', 'optimizer', 'causal', 'supervised_targets_total'):
            if not equal_trees(a[part], b[part]):
                raise AssertionError((variant, 'resume mismatch', part))
        result.append(variant + ': CPU 4-step versus 2+2 resume exact state match')
        print('PASS:', result[-1], flush=True)
        del a, b

    train('base', 'new_base', 4)
    if args.reference_root:
        train('base', 'old_base', 4, entry=Path(args.reference_root)/'train.py')
        a = trusted_load(work/'new_base/checkpoints/resume_000004.pt')
        b = trusted_load(work/'old_base/checkpoints/resume_000004.pt')
        for part in ('model', 'ema', 'optimizer', 'causal'):
            if not equal_trees(a[part], b[part]):
                raise AssertionError(('old/new base mismatch', part))
        result.append('Original MCWP Base versus new reference: exact CPU states after 4 updates')
        print('PASS:', result[-1], flush=True)

    train('monolithic', 'monolithic_resumed', 5, resume='fresh', success=False)
    train('static_fusion', 'monolithic_resumed', 5, resume='auto', success=False)
    result.append('Fresh overwrite and cross-variant resume correctly refused')

    for variant in REQUIRED:
        train(variant, 'ddp_'+variant, 2, world=2)
        train(variant, 'ddp_'+variant, 4, resume='auto', world=2)
        bundle = trusted_load(work/('ddp_'+variant+'/checkpoints/resume_000004.pt'))
        assert len(bundle['rng_by_rank']) == 2
        assert bundle['completed_steps'] == 4
        result.append(variant + ': 2-process CPU/Gloo DDP forward/backward/save/resume passed')
        print('PASS:', result[-1], flush=True)

    train('static_fusion', 'ddp_static_continuous', 4, world=2)
    a = trusted_load(work/'ddp_static_continuous/checkpoints/resume_000004.pt')
    b = trusted_load(work/'ddp_static_fusion/checkpoints/resume_000004.pt')
    for part in ('model', 'ema', 'optimizer', 'causal', 'supervised_targets_total'):
        assert equal_trees(a[part], b[part]), ('DDP exact resume mismatch', part)
    result.append('2-process DDP static_fusion: continuous vs resumed exact CPU states')

    train('horizon1', 'workers_test', 2, workers=2)
    result.append('Spawn DataLoader with 2 workers passed')

    evaluate = [sys.executable, str(ROOT/'evaluate.py'), '--cpu', '--test-root', str(work/'test_cases'),
                '--out-dir', str(work/'evaluation'), '--seeds', '0', '--warmup-rollouts', '0', '--save-predictions']
    for variant in REQUIRED:
        evaluate += ['--checkpoint', str(work/(variant+'_continuous/eval_checkpoints/ema_000004.pt'))]
    if args.reference_root:
        evaluate += ['--checkpoint', str(work/'old_base/eval_checkpoints/ema_000004.pt')]
    execute(evaluate, 'evaluation', timeout=240)
    import csv
    with open(work/'evaluation/metrics.csv') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (5 if args.reference_root else 4)*6
    assert all(row['status'] == 'PASS' for row in rows)
    result.append('Mixed ablation/legacy evaluator: 100-frame rollout, CSV and NPZ passed on synthetic fixtures')
    (work/'RESULTS.json').write_text(json.dumps({'torch': str(torch.__version__), 'cuda': torch.cuda.is_available(), 'tests': result}, indent=2))
    print('\nALL INTEGRATION TESTS PASSED. Results:', work/'RESULTS.json', flush=True)
    if tmp:
        tmp.cleanup()


if __name__ == '__main__':
    main()
