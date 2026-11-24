"""Small compatibility/checkpoint helpers. Only load trusted local checkpoints."""
import hashlib
import inspect
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def trusted_load(path, map_location='cpu'):
    kwargs = {'map_location': map_location}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def cpu_tree(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(cpu_tree(v) for v in obj)
    return obj


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        with open(tmp, 'wb') as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def distributed():
    return dist.is_available() and dist.is_initialized()


def all_gather_tensor(x):
    if not distributed():
        return x
    xs = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)


def get_rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device).cpu() if device.type == 'cuda' else None}


def set_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)


class CausalWeights:
    """GenWP-style running error weights; update once per global FIRST-step batch.

    Equations: w[n]=exp(-epsilon*sum_{k<n}L[k]); L[n] is an error EMA.
    Each rank receives the same ordered global sample list before updating.
    """
    def __init__(self, device, epsilon=0.1, decay=0.9999, delta=0.99):
        self.table = torch.ones(101, device=device)
        self.epsilon, self.decay, self.delta = epsilon, decay, delta

    @torch.no_grad()
    def weights(self):
        prefix = torch.cat((self.table.new_zeros(1), self.table[:-1].cumsum(0)))
        w = torch.exp(-self.epsilon * prefix)
        return torch.where(w >= self.delta, torch.ones_like(w), w)

    @torch.no_grad()
    def update(self, indices, errors):
        for n, error in zip(indices.cpu().tolist(), errors.unbind(0)):
            self.table[n] = self.decay*self.table[n] + (1-self.decay)*error

    def state_dict(self):
        return {'table': self.table.detach().cpu(), 'epsilon': self.epsilon,
                'decay': self.decay, 'delta': self.delta}

    def load_state_dict(self, d):
        for name in ('epsilon', 'decay', 'delta'):
            if getattr(self, name) != d[name]:
                raise ValueError('Causal hyperparameter mismatch: ' + name)
        self.table.copy_(d['table'].to(self.table.device))
