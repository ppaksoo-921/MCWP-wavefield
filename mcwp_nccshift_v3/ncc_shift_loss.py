"""Threshold-free local NCC displacement auxiliary loss (v3).

The original exact-alignment pixel MSE is NOT replaced. This module adds only
an auxiliary pressure when a local mismatch is both (a) explainable by a small
spatial shift and (b) locally correlated after shifting.

For each local window and integer offset delta within radius r:
    rho(delta) = zero-mean normalized cross correlation (NCC)
    explain = clip((rho_best-rho_0)/(1-rho_0+eps), 0, 1)

A threshold-free detached gate is used:
    gate = explain * max(rho_best,0)^2 * energy_gate

The NCC values are converted to a differentiable shift distribution. The loss
penalizes the squared norm of the *expected displacement vector* rather than
expected squared distance. This supplies directional pressure toward zero
shift only when the NCC landscape has a coherent preferred direction; an
ambiguous symmetric landscape largely cancels instead of forcing an arbitrary
move.

The auxiliary is scaled by a detached local exact-alignment residual, so
amplitude/shape errors continue to be learned by the unchanged MSE branch.
Predictions are never aligned before recurrent feedback or evaluation.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class NCCShiftConfig:
    radius: int = 4
    window: int = 16
    stride: int = 16
    temperature: float = 0.07
    eps: float = 1e-6

    def validate(self) -> None:
        if self.radius < 1:
            raise ValueError('radius must be >=1')
        if self.window < 4 or self.stride < 1:
            raise ValueError('invalid window/stride')
        if self.temperature <= 0 or self.eps <= 0:
            raise ValueError('temperature and eps must be positive')

    def to_dict(self) -> Dict:
        return asdict(self)


class NCCShiftAuxLoss(nn.Module):
    """Local NCC-based directional displacement regularizer for [B,H,W]."""
    def __init__(self, config: NCCShiftConfig):
        super().__init__()
        config.validate()
        self.config = config
        r = config.radius
        offsets: List[Tuple[int, int]] = [(0, 0)]
        offsets += sorted(
            [(dz, dx) for dz in range(-r, r + 1) for dx in range(-r, r + 1)
             if (dz, dx) != (0, 0) and dz*dz + dx*dx <= r*r],
            key=lambda q: (q[0]*q[0] + q[1]*q[1], q[0], q[1]))
        self.offsets = offsets
        table = torch.tensor(offsets, dtype=torch.float32)
        self.register_buffer('offset_table', table)
        self.register_buffer('offset_table_normalized', table / float(r))

    def _pool(self, x: Tensor) -> Tensor:
        return F.avg_pool2d(x, kernel_size=self.config.window, stride=self.config.stride)

    def forward(self, prediction: Tensor, reference: Tensor):
        if prediction.ndim != 3 or prediction.shape != reference.shape:
            raise ValueError('expected matching [B,H,W] tensors')
        if prediction.dtype not in (torch.float32, torch.float64):
            raise ValueError('use FP32/FP64')
        if prediction.device != reference.device:
            raise ValueError('device mismatch')

        _, h, w = prediction.shape
        r = self.config.radius
        if h <= 2*r + self.config.window or w <= 2*r + self.config.window:
            raise ValueError('frame too small for selected radius/window')

        # Same target pixels are compared for every candidate; no periodic wrap.
        target = reference[:, None, r:h-r, r:w-r]
        pred0 = prediction[:, None, r:h-r, r:w-r]

        mean_t = self._pool(target)
        mean_t2 = self._pool(target.square())
        var_t = (mean_t2 - mean_t.square()).clamp_min(0)
        local_power = mean_t2.detach().squeeze(1)
        frame_power = reference.square().mean((-2, -1), keepdim=True).detach()

        # Detached local exact-alignment residual is a scale only; it contributes
        # no gradient and therefore does not duplicate the amplitude/shape branch.
        local_residual = self._pool((pred0.detach() - target.detach()).square()).squeeze(1)

        nccs = []
        for dz, dx in self.offsets:
            pred = prediction[:, None, r+dz:h-r+dz, r+dx:w-r+dx]
            mean_p = self._pool(pred)
            mean_p2 = self._pool(pred.square())
            mean_pt = self._pool(pred * target)
            cov = mean_pt - mean_p * mean_t
            var_p = (mean_p2 - mean_p.square()).clamp_min(0)
            denom = torch.sqrt(var_p * var_t + self.config.eps)
            nccs.append((cov / denom).clamp(-1, 1))

        rho = torch.cat(nccs, dim=1)  # [B,S,Gz,Gx]
        rho0 = rho[:, 0]
        rho_best, best_idx = rho.max(dim=1)

        # Threshold-free gate. It is detached so the network cannot reduce the
        # auxiliary merely by worsening NCC. Squared quality aggressively
        # suppresses weak accidental matches such as late low-energy frames.
        explain = ((rho_best - rho0) / (1.0 - rho0 + self.config.eps)).clamp(0, 1).detach()
        quality = rho_best.clamp(0, 1).detach()
        energy_gate = (
            local_power / (local_power + frame_power[:, 0, 0][:, None, None] + self.config.eps)
        ).detach()
        gate = explain * quality.square() * energy_gate

        probs = torch.softmax(rho / self.config.temperature, dim=1)
        offsets_norm = self.offset_table_normalized.to(device=rho.device, dtype=rho.dtype)
        # [B,Gz,Gx,2], normalized by radius.
        expected_vec = torch.einsum('bshw,sc->bhwc', probs, offsets_norm)
        expected_vec_d2 = expected_vec.square().sum(-1)  # 0..~1

        # Local residual only fixes units/magnitude; all correction direction
        # comes from the differentiable NCC displacement distribution.
        per_sample = (gate * expected_vec_d2 * local_residual).mean((-2, -1))
        aux = per_sample.mean()

        with torch.no_grad():
            offsets = self.offset_table.to(device=rho.device, dtype=rho.dtype)
            best_delta = offsets[best_idx]
            best_px = best_delta.square().sum(-1).sqrt()
            expected_vec_px = expected_vec * float(self.config.radius)
            directional_px = expected_vec_px.square().sum(-1).sqrt()
            diag = {
                'ncc_zero': rho0.mean(),
                'ncc_best': rho_best.mean(),
                'explainability': explain.mean(),
                'gate': gate.mean(),
                'best_shift_px': best_px.mean(),
                'directional_shift_px': directional_px.mean(),
                'boundary_fraction': (best_px >= (self.config.radius - 1e-6)).float().mean(),
                'aux_loss': aux.detach(),
            }
        return aux, diag
