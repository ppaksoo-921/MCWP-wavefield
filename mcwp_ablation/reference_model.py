"""Deterministic, medium-conditioned wave propagator. PyTorch >= 1.12.

History is NEWEST FIRST, raw pressure, [B, 5, Z, X]. Velocity is raw m/s.
This is an experimental multiscale CNN + spectral operator, not a CNO replica.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    width: int = 32
    history: int = 5
    modes: int = 8
    spectral_pad: int = 4
    velocity_min: float = 1400.0
    velocity_max: float = 5000.0

    def to_dict(self):
        return asdict(self)


def groups(channels):
    for n in (8, 4, 2, 1):
        if channels % n == 0:
            return n
    return 1


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(groups(cout), cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups(cout), cout)
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride=stride)

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), 16)))
        self.mlp = nn.Sequential(nn.Linear(33, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, target_index):
        # Same available absolute index as GenWP; not the physical time in seconds.
        t = target_index.float().reshape(-1, 1) * 0.001
        phase = t * self.frequencies.reshape(1, -1)
        return self.mlp(torch.cat((t, phase.sin(), phase.cos()), dim=1))


class MediumConditionedBlock(nn.Module):
    """Spatial mixture of local stencils; medium modulates the update operator."""
    def __init__(self, channels, time_width):
        super().__init__()
        self.norm = nn.GroupNorm(groups(channels), channels)
        self.mod = nn.Conv2d(channels, 2 * channels, 1)
        self.gates = nn.Conv2d(channels, 3, 1)
        self.time = nn.Linear(time_width, channels)
        self.stencils = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=d, dilation=d, groups=channels)
            for d in (1, 2, 3)
        ])
        self.mix = nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, wave, medium, time):
        scale, shift = self.mod(medium).chunk(2, dim=1)
        h = self.norm(wave) * (1 + 0.5 * torch.tanh(scale)) + shift
        h = F.silu(h + self.time(time)[:, :, None, None])
        gates = torch.softmax(self.gates(medium), dim=1)
        update = sum(gates[:, k:k+1] * op(h) for k, op in enumerate(self.stencils))
        return wave + 0.1 * self.mix(update)


class SpectralBlock(nn.Module):
    """FNO-style low-mode convolution. All trainable parameters are REAL for DDP."""
    def __init__(self, channels, modes, pad):
        super().__init__()
        self.modes = modes
        self.pad = pad
        scale = 1.0 / channels
        self.top = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.bottom = nn.Parameter(scale * torch.randn(channels, channels, modes, modes, 2))
        self.local = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(groups(channels), channels)

    def forward(self, x):
        h = self.norm(x)
        hpad = F.pad(h, (self.pad,) * 4, mode='replicate') if self.pad else h
        b, c, nz, nx = hpad.shape
        spectrum = torch.fft.rfft2(hpad, norm='ortho')
        out = torch.zeros_like(spectrum)
        mz = min(self.modes, nz // 2)
        mx = min(self.modes, nx // 2 + 1)
        wt = torch.view_as_complex(self.top.contiguous())[:, :, :mz, :mx]
        wb = torch.view_as_complex(self.bottom.contiguous())[:, :, :mz, :mx]
        out[:, :, :mz, :mx] = torch.einsum('bixy,ioxy->boxy', spectrum[:, :, :mz, :mx], wt)
        out[:, :, -mz:, :mx] = torch.einsum('bixy,ioxy->boxy', spectrum[:, :, -mz:, :mx], wb)
        y = torch.fft.irfft2(out, s=(nz, nx), norm='ortho')
        if self.pad:
            y = y[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return x + 0.1 * F.silu(y + self.local(h))


class MediumConditionedWavePropagator(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.history != 5 or self.config.width < 4:
            raise ValueError('This release requires history=5 and width>=4.')
        w = self.config.width
        self.channels = (w, 2*w, 3*w, 4*w)
        tw = 4*w
        self.time_embedding = TimeEmbedding(tw)
        # Two independent pyramids. No velocity concatenation in temporal stem.
        self.medium_stem = ConvBlock(1, w)
        self.medium_down = nn.ModuleList([
            ConvBlock(self.channels[i], self.channels[i+1], stride=2) for i in range(3)
        ])
        self.temporal_stem = ConvBlock(9, w)  # 5 raw frames + 4 differences
        self.wave_down = nn.ModuleList([
            ConvBlock(self.channels[i], self.channels[i+1], stride=2) for i in range(3)
        ])
        self.encoder_interactions = nn.ModuleList([MediumConditionedBlock(c, tw) for c in self.channels])
        self.spectral = SpectralBlock(self.channels[-1], self.config.modes, self.config.spectral_pad)
        self.up = nn.ModuleList([ConvBlock(self.channels[i+1] + self.channels[i], self.channels[i])
                                 for i in (2, 1, 0)])
        self.decoder_interactions = nn.ModuleList([MediumConditionedBlock(self.channels[i], tw)
                                                  for i in (2, 1, 0)])
        self.head = nn.Conv2d(w, 1, 3, padding=1)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)

    def encode_medium(self, vp):
        if vp.ndim == 3:
            vp = vp[:, None]
        c = 2 * (vp - self.config.velocity_min) / (self.config.velocity_max - self.config.velocity_min) - 1
        out = [self.medium_stem(c)]
        for layer in self.medium_down:
            out.append(layer(out[-1]))
        return out

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
        # Initial padded history has no u[-1]; use u[0] at the first prediction.
        valid_second = (target_index.reshape(-1, 1, 1, 1) >= 2).to(history.dtype)
        extrapolated = history[:, :1] + valid_second * (history[:, :1] - history[:, 1:2])
        return extrapolated + correction

    def forward(self, history, vp, target_index, horizon=1):
        """Each step outputs ONE pressure field; optional differentiable rollout."""
        medium = self.encode_medium(vp)
        predicted = []
        buffer = history
        for k in range(horizon):
            nxt = self.predict_next(buffer, target_index + k, medium)
            predicted.append(nxt)
            buffer = torch.cat((nxt, buffer[:, :-1]), dim=1)
        return torch.cat(predicted, dim=1)  # [B,horizon,Z,X]
