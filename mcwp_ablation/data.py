"""Read existing GenWP HDF5 files without modifying them.

Sampling is stateless: a tuple (file_index, target_frame) is predetermined from
(seed, epoch, global position), so workers and prefetching do not change resume.
"""
import hashlib
import random
from collections import OrderedDict
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class WaveDataset(Dataset):
    def __init__(self, root, horizon=3, cache_size=16):
        self.root = Path(root).expanduser().resolve()
        self.files = sorted(self.root.rglob('*.h5'))
        if not self.files:
            raise FileNotFoundError('No .h5 files under ' + str(self.root))
        self.horizon, self.cache_size = horizon, cache_size
        self.cache = OrderedDict()
        records = '\n'.join(str(p.relative_to(self.root)) for p in self.files)
        self.index_hash = hashlib.sha256(records.encode()).hexdigest()

    def __len__(self):
        return len(self.files)

    def close(self):
        for f in self.cache.values():
            f.close()
        self.cache.clear()

    def __getstate__(self):
        d = self.__dict__.copy()
        d['cache'] = OrderedDict()
        return d

    def get_file(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if len(self.cache) >= self.cache_size:
            _, old = self.cache.popitem(last=False)
            old.close()
        f = h5py.File(self.files[idx], 'r')
        self.cache[idx] = f
        return f

    def __getitem__(self, key):
        idx, n = (int(key[0]), int(key[1]))
        try:
            f = self.get_file(idx)
            if f['snaps'].shape != (101, 128, 128) or f['vp'].shape != (128, 128):
                raise ValueError('Expected snaps=(101,128,128), vp=(128,128).')
            if f['snaps'].dtype != np.float32 or f['vp'].dtype != np.float32:
                raise ValueError('Expected float32 arrays.')
            if not 1 <= n <= 100:
                raise ValueError('Target index outside 1..100.')
            begin, end = max(0, n-5), min(101, n+self.horizon)
            frames = f['snaps'][begin:end]
            vp = f['vp'][:]
            if not np.isfinite(frames).all() or not np.isfinite(vp).all() or vp.min() <= 0:
                raise ValueError('Non-finite arrays or nonpositive velocity.')
            history = np.zeros((5, 128, 128), np.float32)
            for j in range(5):
                t = n-1-j
                if t >= 0:
                    history[j] = frames[t-begin]
            target = np.zeros((self.horizon, 128, 128), np.float32)
            valid = np.zeros(self.horizon, np.float32)
            for j in range(self.horizon):
                t = n+j
                if t <= 100:
                    target[j] = frames[t-begin]
                    valid[j] = 1
            return {'history': history, 'vp': vp[None], 'target': target,
                    'valid': valid, 'index': np.int64(n)}
        except Exception as exc:
            raise RuntimeError(f'{self.files[idx]} target={n}: {exc}') from exc


class EpochBatchSampler(Sampler):
    def __init__(self, size, global_batch, rank, world, seed, epoch, offset=0):
        if global_batch % world:
            raise ValueError('global_batch must be divisible by world size.')
        self.size, self.global_batch = size, global_batch
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch, self.offset = epoch, offset
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
            batch = []
            for p in range(start, start+self.local_batch):
                # Stable across Python processes/platform hash randomization.
                text = f'{self.seed}:{self.epoch}:{p}:{order[p]}'
                value = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), 'little')
                n = random.Random(value).randint(1, 100)
                batch.append((order[p], n))
            yield batch


class ResumableLoader:
    def __init__(self, dataset, global_batch, rank, world, seed, workers, step, pin_memory):
        self.dataset, self.global_batch = dataset, global_batch
        self.rank, self.world, self.seed, self.workers = rank, world, seed, workers
        self.pin_memory = pin_memory
        self.steps_per_epoch = len(dataset) // global_batch
        if self.steps_per_epoch == 0:
            raise ValueError('At least one global batch of trajectories required.')
        self.epoch, offset = divmod(step, self.steps_per_epoch)
        self.iterator = None
        self._start(offset)

    def _start(self, offset):
        sampler = EpochBatchSampler(len(self.dataset), self.global_batch, self.rank,
                                    self.world, self.seed, self.epoch, offset)
        kwargs = dict(dataset=self.dataset, batch_sampler=sampler, num_workers=self.workers,
                      pin_memory=self.pin_memory,
                      generator=torch.Generator().manual_seed(self.seed+self.epoch*10000+self.rank))
        if self.workers:
            # Spawn avoids CUDA-after-fork issues; sampling does not use worker RNG.
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
