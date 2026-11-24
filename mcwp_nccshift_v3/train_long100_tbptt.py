"""Long free-rollout fine-tuning with TBPTT and sparse local NCC-shift supervision.

Purpose
-------
Match evaluation more closely than the original short-horizon trainer:
  * always start from frame 0 (history = [u0, 0, 0, 0, 0]);
  * free-roll recursively for --rollout-steps (default 100);
  * backpropagate through only --tbptt-chunk consecutive steps at a time
    (default 20), detaching recurrent history between chunks;
  * keep exact-alignment pixel MSE on EVERY predicted frame;
  * add the existing threshold-free local NCC displacement auxiliary only at
    chunk/end-point frames (20,40,60,80,100 with defaults).

No prediction is shifted before recurrent feedback. Evaluation can therefore
remain completely unchanged and unaligned.

Place this file next to model.py, data.py, common.py and ncc_shift_loss.py in
mcwp_nccshift_v3. It intentionally reuses those files without modifying them.
"""
import argparse
import csv
import fcntl
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler

from common import (CausalWeights, all_gather_tensor, atomic_save, cpu_tree,
                    get_rng, set_rng, trusted_load, hash_file, write_json)
from data import WaveDataset
from model import ModelConfig, MediumConditionedWavePropagator
from ncc_shift_loss import NCCShiftConfig, NCCShiftAuxLoss

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=str(Path.home()/'Wage_surrogate/GenWP-pub/dataset/phaseA_outputs/h5'))
    p.add_argument('--run-dir', required=True)
    p.add_argument('--max-steps', type=int, default=5000,
                   help='Optimizer updates. One update contains one full free rollout per trajectory.')
    p.add_argument('--global-batch', type=int, default=6)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--modes', type=int, default=8)
    p.add_argument('--rollout-steps', type=int, default=100)
    p.add_argument('--tbptt-chunk', type=int, default=20,
                   help='Gradient-connected rollout length before recurrent history is detached.')
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--ema', type=float, default=0.999)
    p.add_argument('--seed', type=int, default=20260909)
    p.add_argument('--no-causal', action='store_true',
                   help='Disable the original causal frame weights. Exact MSE still supervises all rollout frames.')
    p.add_argument('--grad-clip', type=float, default=0.0, help='0 means no clipping.')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--eval-at', default='100,250')
    p.add_argument('--keep-last', type=int, default=2)
    p.add_argument('--preserve-full', default='1000,2000,5000')
    p.add_argument('--resume', default='', help='empty/fresh, auto, or explicit FULL checkpoint path.')
    p.add_argument('--cpu', action='store_true', help='For tiny verification only.')

    # Existing v3 NCC-shift auxiliary; only its scheduling changes here.
    p.add_argument('--shift-window', type=int, default=16)
    p.add_argument('--shift-stride', type=int, default=16)
    p.add_argument('--shift-radius', type=int, default=4)
    p.add_argument('--shift-temperature', type=float, default=0.07)
    p.add_argument('--shift-weight', type=float, default=0.5,
                   help='Additive coefficient. Exact MSE always remains weight 1.')
    p.add_argument('--shift-warmup', type=int, default=250)
    p.add_argument('--shift-decay-start', type=int, default=0)
    p.add_argument('--shift-decay-end', type=int, default=0)
    p.add_argument('--shift-every', type=int, default=1,
                   help='Compute endpoint shift auxiliary every N optimizer updates.')
    p.add_argument('--shift-batch', type=int, default=1,
                   help='Per-rank trajectories used by NCC at each endpoint.')
    p.add_argument('--shift-interval', type=int, default=20,
                   help='Apply NCC auxiliary at rollout frames divisible by this interval.')

    p.add_argument('--checkpoint-label', default='base_long100_tbptt_shift')
    p.add_argument('--init-checkpoint', default='',
                   help='Trusted MCWP EMA/full checkpoint for a NEW fine-tune run.')
    p.add_argument('--init-weights', choices=['ema', 'raw'], default='ema')
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p


def int_set(text):
    return {int(x) for x in text.split(',') if x.strip()}


def broadcast_message(message, world):
    if world > 1:
        payload = [message]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]
    return message


class FixedStartBatchSampler(Sampler):
    """Deterministic trajectory shuffle; every sample starts at target frame 1.

    WaveDataset(horizon=100) with key (file_idx, 1) returns the exact evaluator
    initial state: history[0]=u0 and history[1:]=0, targets=u1..u100.
    """
    def __init__(self, size, global_batch, rank, world, seed, epoch, offset=0):
        if global_batch % world:
            raise ValueError('global_batch must be divisible by world size.')
        self.size = size
        self.global_batch = global_batch
        self.rank = rank
        self.world = world
        self.seed = seed
        self.epoch = epoch
        self.offset = offset
        self.steps = size // global_batch
        self.local_batch = global_batch // world
        if self.steps < 1 or not 0 <= offset <= self.steps:
            raise ValueError('Dataset too small or invalid batch offset.')

    def __len__(self):
        return self.steps - self.offset

    def __iter__(self):
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=gen).tolist()
        for b in range(self.offset, self.steps):
            start = b*self.global_batch + self.rank*self.local_batch
            yield [(order[p], 1) for p in range(start, start+self.local_batch)]


class FixedStartResumableLoader:
    def __init__(self, dataset, global_batch, rank, world, seed, workers, step, pin_memory):
        self.dataset = dataset
        self.global_batch = global_batch
        self.rank = rank
        self.world = world
        self.seed = seed
        self.workers = workers
        self.pin_memory = pin_memory
        self.steps_per_epoch = len(dataset) // global_batch
        if self.steps_per_epoch == 0:
            raise ValueError('At least one global batch of trajectories required.')
        self.epoch, offset = divmod(step, self.steps_per_epoch)
        self.iterator = None
        self._start(offset)

    def _start(self, offset):
        sampler = FixedStartBatchSampler(len(self.dataset), self.global_batch, self.rank,
                                         self.world, self.seed, self.epoch, offset)
        kwargs = dict(dataset=self.dataset, batch_sampler=sampler, num_workers=self.workers,
                      pin_memory=self.pin_memory,
                      generator=torch.Generator().manual_seed(self.seed+self.epoch*10000+self.rank))
        if self.workers:
            kwargs.update(prefetch_factor=2, multiprocessing_context='spawn')
        self.loader = DataLoader(**kwargs)
        self.iterator = iter(self.loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self._start(0)
            return next(self.iterator)

    def close(self):
        self.iterator = None
        self.loader = None
        self.dataset.close()


def main():
    global STOP_REQUESTED
    args = parser().parse_args()

    if args.rollout_steps < 1 or args.rollout_steps > 100:
        raise ValueError('rollout-steps must be within 1..100 for the 101-frame dataset.')
    if args.tbptt_chunk < 1 or args.rollout_steps % args.tbptt_chunk:
        raise ValueError('tbptt-chunk must divide rollout-steps exactly.')
    if args.shift_interval < 1 or args.rollout_steps % args.shift_interval:
        raise ValueError('shift-interval must divide rollout-steps exactly.')
    if args.shift_interval % args.tbptt_chunk:
        raise ValueError('For this one-file trainer, shift-interval must be a multiple of tbptt-chunk. '
                         'Use chunk=20, interval=20 (or chunk=10, interval=10).')
    if args.shift_weight < 0 or args.shift_warmup < 0 or args.shift_every < 1 or args.shift_batch < 1:
        raise ValueError('Invalid shift arguments.')
    if (args.shift_decay_start == 0) != (args.shift_decay_end == 0):
        raise ValueError('Set both shift-decay-start/end, or both to 0.')
    if args.shift_decay_start and not (0 < args.shift_decay_start < args.shift_decay_end):
        raise ValueError('shift-decay-end must be greater than shift-decay-start > 0.')
    if min(args.max_steps, args.global_batch, args.log_every, args.save_every,
           args.eval_every, args.keep_last) < 1:
        raise ValueError('Step intervals, batch and retention must be positive.')
    checkpoint_label = args.checkpoint_label
    if not checkpoint_label or any(not (c.isalnum() or c in '_-') for c in checkpoint_label):
        raise ValueError('checkpoint-label must contain only letters, digits, underscores or hyphens.')

    shift_config = NCCShiftConfig(radius=args.shift_radius, window=args.shift_window,
                                  stride=args.shift_stride, temperature=args.shift_temperature)
    shift_config.validate()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if args.global_batch % world:
        raise ValueError(f'Global batch {args.global_batch} must be evenly divisible by GPU count {world}.')
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    device = torch.device('cpu' if args.cpu else f'cuda:{local}')
    if device.type == 'cuda':
        torch.cuda.set_device(local)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        dist.init_process_group('gloo' if args.cpu else 'nccl', timeout=timedelta(minutes=30))

    run = Path(args.run_dir).expanduser().resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = None
    lock_error = None
    if rank == 0:
        try:
            lock = open(run/'.train.lock', 'a+')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception as e:
            lock_error = f'Another training process owns {run}, or lock failed: {e}'
    lock_error = broadcast_message(lock_error, world)
    if lock_error:
        raise RuntimeError(lock_error)

    data = None
    try:
        full_dir, eval_dir = run/'checkpoints', run/'eval_checkpoints'
        full_dir.mkdir(exist_ok=True)
        eval_dir.mkdir(exist_ok=True)

        resume_path = None
        if rank == 0:
            if args.resume == 'auto':
                candidates = sorted(full_dir.glob('resume_*.pt'))
                resume_path = str(candidates[-1]) if candidates else None
            elif args.resume not in ('', 'fresh'):
                resume_path = str(Path(args.resume).expanduser().resolve())
        resume_path = broadcast_message(resume_path, world)
        if resume_path is None and (list(full_dir.glob('resume_*.pt')) or list(eval_dir.glob('ema_*.pt'))):
            raise RuntimeError('Run already has checkpoints. Use --resume auto or a NEW run directory.')

        bundle = trusted_load(resume_path) if resume_path else None
        if bundle and bundle.get('kind') != 'mcwp_full':
            raise ValueError('Resume requires a full MCWP checkpoint.')
        completed = int(bundle['completed_steps']) if bundle else 0
        if completed >= args.max_steps:
            if rank == 0:
                print(f'Already at {completed} updates; target={args.max_steps}. No work.', flush=True)
            return

        model_config = ModelConfig(width=args.width, modes=args.modes)
        initial = None
        if bundle:
            if bundle['signature'].get('trainer') != 'mcwp_long100_tbptt_v1':
                raise ValueError('Only this trainer can resume this run.')
            if args.init_checkpoint:
                raise ValueError('Do not pass --init-checkpoint when resuming.')
            initialization = bundle['signature'].get('initialization')
        elif args.init_checkpoint:
            init_path = Path(args.init_checkpoint).expanduser().resolve()
            initial = trusted_load(init_path)
            if initial.get('kind') not in ('mcwp_ema', 'mcwp_full'):
                raise ValueError('Initialization requires MCWP EMA/full weights.')
            if initial['model_config'] != model_config.to_dict():
                raise ValueError('Initialization architecture mismatch.')
            if initial['kind'] == 'mcwp_ema' and args.init_weights != 'ema':
                raise ValueError('EMA-only checkpoint has no raw weights.')
            initialization = {'checkpoint': str(init_path), 'sha256': hash_file(init_path),
                              'kind': initial['kind'], 'weights': args.init_weights,
                              'source_completed_steps': int(initial['completed_steps']),
                              'source_variant': initial['variant'],
                              'state_policy': 'weights only; NEW optimizer/EMA/causal/sampler'}
        else:
            initialization = None

        signature = {
            'trainer': 'mcwp_long100_tbptt_v1',
            'initialization': initialization,
            'checkpoint_label': checkpoint_label,
            'model': model_config.to_dict(),
            'global_batch': args.global_batch,
            'world_size': world,
            'rollout_steps': args.rollout_steps,
            'tbptt_chunk': args.tbptt_chunk,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'ema': args.ema,
            'seed': args.seed,
            'causal': not args.no_causal,
            'shift_config': shift_config.to_dict(),
            'shift_weight': args.shift_weight,
            'shift_warmup': args.shift_warmup,
            'shift_decay_start': args.shift_decay_start,
            'shift_decay_end': args.shift_decay_end,
            'shift_every': args.shift_every,
            'shift_batch': args.shift_batch,
            'shift_interval': args.shift_interval,
            'precision': 'float32',
            'tf32': False,
        }

        dataset = WaveDataset(args.data_dir, horizon=args.rollout_steps)
        signature['dataset_index_hash'] = dataset.index_hash
        signature['dataset_count'] = len(dataset)
        manifest = Path(args.data_dir).expanduser().resolve().parents[1]/'phaseA_inputs/phaseA_manifest.csv'
        signature['source_manifest_sha256'] = hash_file(manifest) if manifest.exists() else None
        signature['implementation_sha256'] = {
            n: hash_file(Path(__file__).parent/n)
            for n in ('model.py', 'data.py', 'common.py', 'ncc_shift_loss.py', Path(__file__).name)
        }
        if bundle and bundle['signature'] != signature:
            differing = [k for k in signature if signature[k] != bundle['signature'].get(k)]
            raise ValueError('Resume configuration/data/code mismatch: ' + ', '.join(differing))

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        raw = MediumConditionedWavePropagator(model_config).to(device)
        if bundle:
            raw.load_state_dict(bundle['model'], strict=True)
        elif initial is not None:
            state = initial['ema'] if initial['kind'] == 'mcwp_full' and args.init_weights == 'ema' else initial['model']
            raw.load_state_dict(state, strict=True)
            del state
        del initial

        model = (DistributedDataParallel(raw, device_ids=[local] if device.type == 'cuda' else None,
                                         broadcast_buffers=False, find_unused_parameters=False)
                 if world > 1 else raw)
        opt = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        ema = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        causal = CausalWeights(device)
        shift_loss = NCCShiftAuxLoss(shift_config).to(device)
        if bundle:
            opt.load_state_dict(bundle['optimizer'])
            for name, value in bundle['ema'].items():
                ema[name].copy_(value.to(device))
            causal.load_state_dict(bundle['causal'])

        data = FixedStartResumableLoader(dataset, args.global_batch, rank, world, args.seed,
                                         args.workers, completed, device.type == 'cuda')
        random.seed(args.seed+rank+1)
        np.random.seed(args.seed+rank+1)
        torch.manual_seed(args.seed+rank+1)
        if bundle:
            set_rng(bundle['rng_by_rank'][rank], device)
        del bundle

        if rank == 0:
            version = {'torch': torch.__version__, 'python': sys.version, 'numpy': np.__version__,
                       'device': str(device),
                       'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}
            if not (run/'config.json').exists():
                write_json({'signature': signature, 'runtime': version, 'argv': sys.argv,
                            'created': datetime.now().isoformat()}, run/'config.json')
            endpoints = list(range(args.shift_interval, args.rollout_steps+1, args.shift_interval))
            print(f'Parameters={sum(p.numel() for p in raw.parameters()):,}', flush=True)
            print(f'FULL FREE ROLLOUT={args.rollout_steps} | TBPTT chunk={args.tbptt_chunk} | '
                  f'shift endpoints={endpoints}', flush=True)
            print('Exact MSE supervises EVERY rollout frame. NCC shift is additive only at endpoints; '
                  'recurrent predictions are NEVER aligned.', flush=True)
            print(f'World={world} | local_batch={args.global_batch//world} | global_batch={args.global_batch} | FP32', flush=True)
            print(f'Trajectories={len(dataset):,} | steps/epoch={data.steps_per_epoch}', flush=True)
            if initialization:
                print(f'WEIGHTS-ONLY INITIALIZATION: {initialization}', flush=True)
            print(f'Start updates={completed} | target={args.max_steps} | run={run}', flush=True)

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        full_keep = int_set(args.preserve_full)
        eval_at = int_set(args.eval_at)
        metric_path = run/'metrics.csv'
        fields = ['step','samples','total_loss','raw_data_loss','mse_one_step','mse_rollout',
                  'shift_aux_loss','shift_strength','shift_branch_fraction','ncc_zero','ncc_best',
                  'shift_explainability','shift_gate','best_shift_px','directional_shift_px',
                  'shift_boundary_fraction','shift_endpoint_mean','grad_norm','causal_weight_min',
                  'seconds_per_step','max_peak_allocated_gib']
        sums = np.zeros(len(fields)-2, dtype=np.float64)  # excludes step,samples
        intervals = 0
        clock = time.monotonic()
        started = clock
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        raw.train()

        def save(full, evaluation):
            rngs = None
            if full:
                local_rng = get_rng(device)
                if world > 1:
                    rngs = [None]*world
                    dist.all_gather_object(rngs, local_rng)
                else:
                    rngs = [local_rng]
            error = None
            if rank == 0:
                try:
                    meta = {'format_version': 1, 'completed_steps': completed,
                            'model_config': model_config.to_dict(), 'variant': checkpoint_label,
                            'training_variant': 'base', 'signature': signature}
                    if full:
                        checkpoint = dict(meta, kind='mcwp_full', model=cpu_tree(raw.state_dict()),
                                          ema=cpu_tree(ema), optimizer=cpu_tree(opt.state_dict()),
                                          causal=cpu_tree(causal.state_dict()), rng_by_rank=rngs,
                                          sampler_epoch=completed//data.steps_per_epoch,
                                          sampler_batch_offset=completed%data.steps_per_epoch)
                        dst = full_dir/f'resume_{completed:06d}.pt'
                        atomic_save(checkpoint, dst)
                        del checkpoint
                        print('Saved full: ' + str(dst), flush=True)
                        files = sorted(full_dir.glob('resume_*.pt'))
                        newest = set(files[-args.keep_last:])
                        for path in files:
                            step = int(path.stem.split('_')[-1])
                            if path not in newest and step not in full_keep:
                                path.unlink()
                    if evaluation:
                        dst = eval_dir/f'ema_{completed:06d}.pt'
                        atomic_save(dict(meta, kind='mcwp_ema', model=cpu_tree(ema)), dst)
                        print('Saved EMA: ' + str(dst), flush=True)
                except Exception as e:
                    error = f'Checkpoint save failed: {e}'
            error = broadcast_message(error, world)
            if error:
                raise RuntimeError(error)

        n_chunks = args.rollout_steps // args.tbptt_chunk
        n_shift_points = args.rollout_steps // args.shift_interval

        while completed < args.max_steps:
            batch = next(data)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Fixed-start loader guarantees index==1 and all rollout targets valid.
            if not bool((batch['index'] == 1).all().item()):
                raise RuntimeError('Fixed-start loader invariant violated.')

            update = completed + 1
            warm = min(1.0, update / args.shift_warmup) if args.shift_warmup else 1.0
            decay = 1.0
            if args.shift_decay_start:
                if update >= args.shift_decay_end:
                    decay = 0.0
                elif update > args.shift_decay_start:
                    decay = (args.shift_decay_end-update)/float(args.shift_decay_end-args.shift_decay_start)
            shift_strength = args.shift_weight * warm * decay
            shift_branch_on = (update % args.shift_every == 0) and shift_strength > 0

            opt.zero_grad(set_to_none=True)
            history = batch['history']
            table = causal.weights() if not args.no_causal else torch.ones(101, device=device)

            raw_loss_det = history.new_zeros(())
            shift_loss_det = history.new_zeros(())
            first_mse_det = history.new_zeros(())
            rollout_mse_det = history.new_zeros(())
            shift_diag_sum = {name: history.new_zeros(()) for name in
                              ('ncc_zero','ncc_best','explainability','gate','best_shift_px',
                               'directional_shift_px','boundary_fraction')}
            active_shift_endpoints = 0
            active_shift_endpoint_sum = 0.0
            per_frame_errors = []

            for chunk_id, start in enumerate(range(0, args.rollout_steps, args.tbptt_chunk)):
                length = min(args.tbptt_chunk, args.rollout_steps-start)
                target_index = batch['index'] + start
                sync_context = (model.no_sync() if isinstance(model, DistributedDataParallel)
                                and chunk_id < n_chunks-1 else nullcontext())
                with sync_context:
                    prediction = model(history, batch['vp'], target_index, length)
                    reference = batch['target'][:, start:start+length]
                    squared = (prediction-reference).square().mean(dim=(-2,-1))  # [B,K]
                    abs_indices = torch.arange(start+1, start+length+1, device=device)
                    frame_weights = table[abs_indices][None]
                    # Exact MSE on EVERY frame; divide by full rollout length so
                    # summing chunk gradients equals one 100-frame average loss.
                    raw_chunk = (squared * frame_weights).sum(dim=1).mean() / float(args.rollout_steps)
                    chunk_loss = raw_chunk

                    end_frame = start + length
                    aux = prediction.new_zeros(())
                    if shift_branch_on and end_frame % args.shift_interval == 0:
                        eligible_n = min(args.shift_batch, prediction.shape[0])
                        if eligible_n > 0:
                            aux, diag = shift_loss(prediction[:eligible_n, -1], reference[:eligible_n, -1])
                            # Equal average over 20/40/60/80/100 endpoint auxiliaries.
                            chunk_loss = chunk_loss + shift_strength * aux / float(n_shift_points)
                            for name in shift_diag_sum:
                                shift_diag_sum[name] += diag[name].detach()
                            active_shift_endpoints += 1
                            active_shift_endpoint_sum += float(end_frame)
                            shift_loss_det += aux.detach() / float(n_shift_points)

                    chunk_loss.backward()

                # Diagnostics; do not retain old chunk graph.
                raw_loss_det += raw_chunk.detach()
                rollout_mse_det += squared.detach().sum() / float(squared.shape[0]*args.rollout_steps)
                if start == 0:
                    first_mse_det = squared[:,0].detach().mean()
                per_frame_errors.append(squared.detach())

                # Continue the SAME free rollout, but truncate gradients here.
                # prediction is chronological; recurrent history must be newest-first.
                if length >= 5:
                    history = prediction[:, -5:].flip(1).detach()
                else:
                    combined = torch.cat((prediction, history[:, :5-length]), dim=1)
                    history = combined[:, -5:].flip(1).detach()

            grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(),
                                                       args.grad_clip if args.grad_clip > 0 else float('inf'))
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(f'Non-finite gradients at update {update}.')
            opt.step()

            with torch.no_grad():
                for k, v in raw.state_dict().items():
                    if v.is_floating_point():
                        ema[k].mul_(args.ema).add_(v, alpha=1-args.ema)
                    else:
                        ema[k].copy_(v)
                if not args.no_causal:
                    # Extend the original running causal table to the full free rollout:
                    # every absolute frame gets its own observed prediction error.
                    errors = torch.cat(per_frame_errors, dim=1)  # [B,rollout]
                    frame_ids = torch.arange(1, args.rollout_steps+1, device=device)[None].expand(errors.shape[0], -1)
                    ids_global = all_gather_tensor(frame_ids.reshape(-1))
                    err_global = all_gather_tensor(errors.reshape(-1))
                    causal.update(ids_global, err_global)

            completed += 1

            if active_shift_endpoints:
                for name in shift_diag_sum:
                    shift_diag_sum[name] /= float(active_shift_endpoints)
                shift_endpoint_mean = active_shift_endpoint_sum / float(active_shift_endpoints)
            else:
                shift_endpoint_mean = 0.0
            shift_branch_fraction = active_shift_endpoints / float(n_shift_points)
            total_det = raw_loss_det + shift_strength * shift_loss_det

            stat_values = [
                total_det, raw_loss_det, first_mse_det, rollout_mse_det,
                shift_loss_det, history.new_tensor(shift_strength), history.new_tensor(shift_branch_fraction),
                shift_diag_sum['ncc_zero'], shift_diag_sum['ncc_best'], shift_diag_sum['explainability'],
                shift_diag_sum['gate'], shift_diag_sum['best_shift_px'], shift_diag_sum['directional_shift_px'],
                shift_diag_sum['boundary_fraction'], history.new_tensor(shift_endpoint_mean), grad_norm.detach(),
                history.new_tensor(causal.weights().min().item() if not args.no_causal else 1.0),
            ]
            stats = torch.stack(stat_values)
            if world > 1:
                dist.all_reduce(stats)
                stats /= world
            sums[:len(stats)] += stats.cpu().numpy()
            intervals += 1

            stop = torch.tensor(int(STOP_REQUESTED), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            final = completed == args.max_steps or bool(stop.item())

            if completed % args.log_every == 0 or final:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                now = time.monotonic()
                peak = torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3 if device.type == 'cuda' else 0.0,
                                    device=device)
                if world > 1:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                avg = sums / intervals
                sec_per_step = (now-clock)/intervals
                if rank == 0:
                    row = [completed, completed*args.global_batch, *avg[:17], sec_per_step, peak.item()]
                    header = not metric_path.exists()
                    with open(metric_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if header:
                            writer.writerow(fields)
                        writer.writerow(row)
                    print(
                        f'Step {completed}/{args.max_steps} | raw={avg[1]:.6g} | one={avg[2]:.6g} '
                        f'| rollout={avg[3]:.6g} | shift_aux={avg[4]:.6g} | shift_w={avg[5]:.4g} '
                        f'| explain={avg[9]:.3f} | gate={avg[10]:.5f} | best_shift={avg[11]:.3f}px '
                        f'| dir_shift={avg[12]:.3f}px | boundary={avg[13]:.3f} '
                        f'| shift_frame={avg[14]:.1f} | grad={avg[15]:.4g} '
                        f'| sec/step={sec_per_step:.3f} | peak={peak.item():.2f} GiB', flush=True)
                sums[:] = 0
                intervals = 0
                clock = now

            do_full = completed % args.save_every == 0 or final
            do_eval = completed % args.eval_every == 0 or completed in eval_at or final
            if do_full or do_eval:
                save(do_full, do_eval)
            if final:
                break

        if rank == 0:
            print(f'Training {"stopped on signal" if STOP_REQUESTED else "finished"} at {completed} updates. '
                  f'Elapsed seconds: {time.monotonic()-started:.2f}', flush=True)
    finally:
        if data is not None:
            data.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    main()
