"""Controlled ablations of the supplied 2026-09-09 MCWP implementation.

The reference model is vendored byte-for-byte in reference_model.py. History
is NEWEST FIRST, pressure is unnormalized, velocity is in m/s. Architectures
are intentionally deterministic. 'Static' means learned weights fixed across
inputs, not an untrained or velocity-blind model.
"""
import copy
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from reference_model import (ModelConfig, MediumConditionedWavePropagator,
                             ConvBlock, groups)


PRESETS = {
    'base': dict(architecture='full', horizon=3, update='second_order', required=False),
    'monolithic': dict(architecture='monolithic', horizon=3, update='second_order', required=True),
    'static_fusion': dict(architecture='static_fusion', horizon=3, update='second_order', required=True),
    'horizon1': dict(architecture='full', horizon=1, update='second_order', required=True),
    'first_order': dict(architecture='full', horizon=3, update='first_order', required=True),
    # No new architecture: completes the joint/separated x H1/H3 training grid.
    'monolithic_h1': dict(architecture='monolithic', horizon=1, update='second_order', required=False),
}
REQUIRED = ('monolithic', 'static_fusion', 'horizon1', 'first_order')
MONOLITHIC_MIX_EXPANSION = 2.28


def preset(name):
    if name not in PRESETS:
        raise ValueError('Unknown variant: ' + str(name))
    return dict(PRESETS[name])


class StaticFusionBlock(nn.Module):
    """Separate encoders + concat/ordinary convolution, no input-dependent gates.

    The three dilated depthwise stencils are retained. Their mixing weights
    are global learned constants, shared over batch, location and medium.
    Shared layers copy the reference initialization. No unused dummy weights.
    """
    def __init__(self, original, channels):
        super().__init__()
        self.norm = copy.deepcopy(original.norm)
        self.fuse = nn.Conv2d(2 * channels, channels, 1)
        self.time = copy.deepcopy(original.time)
        self.stencils = copy.deepcopy(original.stencils)
        self.logits = nn.Parameter(torch.zeros(3))
        self.mix = copy.deepcopy(original.mix)

    def forward(self, wave, medium, time):
        h = self.fuse(torch.cat((self.norm(wave), medium), dim=1))
        h = F.silu(h + self.time(time)[:, :, None, None])
        weights = self.logits.softmax(dim=0)
        update = sum(weights[k] * op(h) for k, op in enumerate(self.stencils))
        return wave + 0.1 * self.mix(update)


class StaticWaveBlock(nn.Module):
    """Single-stream static block for the early-fusion U-Net.

    The mix bottleneck is expanded to compensate for removing the medium
    encoder. This is an actively used ordinary convolutional capacity match,
    not an exact topology/FLOP match and not unused parameter padding.
    """
    def __init__(self, original, channels):
        super().__init__()
        self.norm = copy.deepcopy(original.norm)
        self.time = copy.deepcopy(original.time)
        self.stencils = copy.deepcopy(original.stencils)
        self.logits = nn.Parameter(torch.zeros(3))
        hidden = max(1, int(round(MONOLITHIC_MIX_EXPANSION * channels)))
        self.mix = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.SiLU(),
                                 nn.Conv2d(hidden, channels, 3, padding=1))

    def forward(self, wave, medium_unused, time):
        h = F.silu(self.norm(wave) + self.time(time)[:, :, None, None])
        weights = self.logits.softmax(dim=0)
        update = sum(weights[k] * op(h) for k, op in enumerate(self.stencils))
        return wave + 0.1 * self.mix(update)


class StaticFusionPropagator(MediumConditionedWavePropagator):
    def __init__(self, config=None):
        super().__init__(config)
        # Keep RNG outside replacement construction identical to reference.
        # All unmodified layers retain exactly the reference initialization.
        with torch.random.fork_rng(devices=[]):
            self.encoder_interactions = nn.ModuleList([
                StaticFusionBlock(old, c) for old, c in zip(self.encoder_interactions, self.channels)
            ])
            self.decoder_interactions = nn.ModuleList([
                StaticFusionBlock(old, self.channels[i])
                for old, i in zip(self.decoder_interactions, (2, 1, 0))
            ])


class FirstOrderPropagator(MediumConditionedWavePropagator):
    """Change only the explicit output prior, not the five-frame history.

    u[n] = u[n-1] + correction. This is NOT a first-order acoustic PDE solver.
    """
    def predict_next(self, history, target_index, medium_features):
        if history.ndim != 4 or history.shape[1] != 5:
            raise ValueError('Expected newest-first history [B,5,Z,X].')
        emb = self.time_embedding(target_index)
        temporal = torch.cat((history, history[:, :-1] - history[:, 1:]), dim=1)
        x = self.temporal_stem(temporal)
        skips = []
        for i in range(4):
            if i:
                x = self.wave_down[i-1](x)
            x = self.encoder_interactions[i](x, medium_features[i], emb)
            skips.append(x)
        x = self.spectral(x)
        for j, i in enumerate((2, 1, 0)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.up[j](torch.cat((x, skips[i]), dim=1))
            x = self.decoder_interactions[j](x, medium_features[i], emb)
        correction = self.head(x)
        return history[:, :1] + correction


class MonolithicPropagator(MediumConditionedWavePropagator):
    """One early-fusion encoder: [5 frames, 4 differences, normalized velocity].

    Same pyramid output widths, time embedding, spectral block, decoder,
    stencil receptive fields and second-order output form as reference.
    Removing a whole branch necessarily changes capacity allocation/depth;
    this control is parameter matched, not a mathematically single-edge edit.
    """
    def __init__(self, config=None):
        super().__init__(config)
        w = self.config.width
        with torch.random.fork_rng(devices=[]):
            original = self.temporal_stem
            joint = ConvBlock(10, w)
            # Reuse all unchanged weights; preserve the nine temporal channels.
            original_state = original.state_dict()
            for name, tensor in joint.state_dict().items():
                source = original_state.get(name)
                if source is not None and tensor.shape == source.shape:
                    tensor.copy_(source)
            joint.conv1.weight.data[:, :9].copy_(original.conv1.weight.data)
            joint.skip.weight.data[:, :9].copy_(original.skip.weight.data)
            self.joint_stem = joint
            self.encoder_interactions = nn.ModuleList([
                StaticWaveBlock(old, c) for old, c in zip(self.encoder_interactions, self.channels)
            ])
            self.decoder_interactions = nn.ModuleList([
                StaticWaveBlock(old, self.channels[i])
                for old, i in zip(self.decoder_interactions, (2, 1, 0))
            ])
        # Delete all unused modules so DDP and parameter counts are truthful.
        del self.temporal_stem
        del self.medium_stem
        del self.medium_down

    def encode_medium(self, vp):
        # This caches only raw normalization. There is no separate medium encoder.
        if vp.ndim == 3:
            vp = vp[:, None]
        return 2 * (vp - self.config.velocity_min) / (self.config.velocity_max - self.config.velocity_min) - 1

    def predict_next(self, history, target_index, medium_features):
        if history.ndim != 4 or history.shape[1] != 5:
            raise ValueError('Expected newest-first history [B,5,Z,X].')
        emb = self.time_embedding(target_index)
        temporal = torch.cat((history, history[:, :-1] - history[:, 1:]), dim=1)
        x = self.joint_stem(torch.cat((temporal, medium_features), dim=1))
        skips = []
        for i in range(4):
            if i:
                x = self.wave_down[i-1](x)
            x = self.encoder_interactions[i](x, None, emb)
            skips.append(x)
        x = self.spectral(x)
        for j, i in enumerate((2, 1, 0)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode='bilinear', align_corners=False)
            x = self.up[j](torch.cat((x, skips[i]), dim=1))
            x = self.decoder_interactions[j](x, None, emb)
        correction = self.head(x)
        valid = (target_index.reshape(-1, 1, 1, 1) >= 2).to(history.dtype)
        return history[:, :1] + valid * (history[:, :1] - history[:, 1:2]) + correction


def build_model(config=None, variant='base'):
    definition = preset(variant)
    config = config or ModelConfig()
    if config.width < 4 or config.history != 5 or config.modes < 1:
        raise ValueError('Require width>=4, history=5 and modes>=1.')
    if definition['architecture'] == 'monolithic':
        return MonolithicPropagator(config)
    if definition['architecture'] == 'static_fusion':
        return StaticFusionPropagator(config)
    if definition['update'] == 'first_order':
        return FirstOrderPropagator(config)
    return MediumConditionedWavePropagator(config)


def parameter_table(width=32, modes=8):
    result = []
    baseline = None
    for name in PRESETS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260909)
            model = build_model(ModelConfig(width=width, modes=modes), name)
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if baseline is None:
            baseline = count
        result.append(dict(variant=name, parameters=count, difference_pct=100 * (count/baseline - 1), **preset(name)))
    return result


if __name__ == '__main__':
    for row in parameter_table():
        print(f"{row['variant']:16s} {row['parameters']:>10,d}  {row['difference_pct']:+.4f}%  H={row['horizon']}")
