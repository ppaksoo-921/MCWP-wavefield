"""Algebra-preserving caching for the released MCWP architecture."""
import torch
from torch.nn import functional as F


def prepare_cache(model, medium, count):
    # All of this runs INSIDE every measured trajectory, never across shots.
    targets = torch.arange(1, count + 1, device=medium[0].device, dtype=torch.long)
    embeddings = model.time_embedding(targets)

    def prepare(block, features):
        scale, shift = block.mod(features).chunk(2, dim=1)
        gain = 1 + 0.5 * torch.tanh(scale)
        gates = torch.softmax(block.gates(features), dim=1)
        projected_times = block.time(embeddings)
        return gain, shift, gates, projected_times

    enc = [prepare(b, medium[i]) for i, b in enumerate(model.encoder_interactions)]
    dec = [prepare(b, medium[i]) for b, i in zip(model.decoder_interactions, (2, 1, 0))]
    valid_second = (targets >= 2).to(medium[0].dtype).reshape(-1, 1, 1, 1)
    return enc, dec, valid_second


def cached_block(block, wave, saved, step):
    gain, shift, gates, projected_times = saved
    h = block.norm(wave) * gain + shift
    h = F.silu(h + projected_times[step:step+1, :, None, None])
    update = sum(gates[:, k:k+1] * op(h) for k, op in enumerate(block.stencils))
    return wave + 0.1 * block.mix(update)


def cached_next(model, history, cache, step):
    enc, dec, valid_second = cache
    temporal = torch.cat((history, history[:, :-1] - history[:, 1:]), dim=1)
    x = model.temporal_stem(temporal)
    skips = []
    for i in range(4):
        if i:
            x = model.wave_down[i-1](x)
        x = cached_block(model.encoder_interactions[i], x, enc[i], step)
        skips.append(x)
    x = model.spectral(x)
    for j, i in enumerate((2, 1, 0)):
        x = F.interpolate(x, size=skips[i].shape[-2:], mode='bilinear', align_corners=False)
        x = model.up[j](torch.cat((x, skips[i]), dim=1))
        x = cached_block(model.decoder_interactions[j], x, dec[j], step)
    return (history[:, :1] + valid_second[step:step+1] *
            (history[:, :1] - history[:, 1:2]) + model.head(x))


@torch.no_grad()
def predict(model, vp_np, ref_np, device, cached=False):
    """Native and cached paths share input/output preparation and history updates."""
    vp = torch.as_tensor(vp_np, dtype=torch.float32, device=device)[None, None]
    u0 = torch.as_tensor(ref_np[0], dtype=torch.float32, device=device)
    medium = model.encode_medium(vp)
    count = len(ref_np) - 1
    cache = prepare_cache(model, medium, count) if cached else None
    history = torch.zeros((1, 5, *vp_np.shape), dtype=torch.float32, device=device)
    history[:, 0] = u0
    out = torch.empty((1, count + 1, *vp_np.shape), dtype=torch.float32, device=device)
    out[:, 0] = u0
    for j in range(count):
        if cached:
            nxt = cached_next(model, history, cache, j)
        else:
            # Same native path as the paper's original evaluator.
            target = torch.tensor([j + 1], dtype=torch.long, device=device)
            nxt = model.predict_next(history, target, medium)
        out[:, j+1:j+2] = nxt
        history = torch.cat((nxt, history[:, :-1]), dim=1)
    return out[0]
