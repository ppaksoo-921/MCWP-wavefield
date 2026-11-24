"""FD-reference-calibrated acoustic residual consistency, NOT zero-residual PINN.

R_h(p) = p_next - 2*p + p_prev - dt_snapshot**2 * c**2 * Laplacian_h(p).
Loss compares R_h(predicted block) with R_h(reference block). This cancels the
reference residual caused by coarse temporal sampling, forcing and FD mismatch.
The loss is supervised, requires FD labels, and is not an independent PDE solver.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AcousticConsistencyLoss(nn.Module):
    def __init__(self, dx=10.0, dz=10.0, snapshot_dt=0.01, margin=12):
        super().__init__()
        if min(dx, dz, snapshot_dt) <= 0 or margin < 2:
            raise ValueError('Positive spacings and margin>=2 required.')
        self.dx, self.dz, self.dt, self.margin = dx, dz, snapshot_dt, margin
        k = torch.zeros(1, 1, 5, 5)
        k[0, 0, 2, :] += torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12]) / dx**2
        k[0, 0, :, 2] += torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12]) / dz**2
        self.register_buffer('kernel', k)

    def residual(self, sequence, vp):
        b, length, nz, nx = sequence.shape
        center = sequence[:, 1:-1]
        lap = F.conv2d(center.reshape(-1, 1, nz, nx), self.kernel, padding=2).reshape(b, length-2, nz, nx)
        return sequence[:, 2:] - 2*center + sequence[:, :-2] - self.dt**2 * vp**2 * lap

    def forward(self, prediction, reference, previous, vp, valid):
        """Return per-sample loss and detached raw-reference residual diagnostic.

        prediction/reference: [B,H,Z,X], H>=2. previous: [B,1,Z,X].
        vp: [B,1,Z,X], raw m/s. valid: [B,H], excludes padded targets >100.
        No reference future field is supplied to the neural model.
        """
        if prediction.shape != reference.shape or prediction.shape[1] < 2:
            raise ValueError('Physics consistency requires matching blocks of >=2 predictions.')
        if min(prediction.shape[-2:]) <= 2*self.margin:
            raise ValueError('Physics margin too large for grid.')
        # Evaluate R(pred)-R(ref) directly on errors for numerical stability.
        error = prediction - reference
        error_seq = torch.cat((torch.zeros_like(previous), error), dim=1)
        residual_error = self.residual(error_seq, vp)
        # Absolute row-sum scale: pressure units, less sensitivity to c and grid.
        scale = 4.0 + self.dt**2 * vp**2 * (16.0/3.0) * (1/self.dx**2 + 1/self.dz**2)
        m = self.margin
        normalized = (residual_error / scale)[:, :, m:-m, m:-m]
        center_valid = valid[:, :-1] * valid[:, 1:]
        per_center = normalized.square().mean(dim=(-2, -1))
        loss = (per_center * center_valid).sum(dim=1) / center_valid.sum(dim=1).clamp_min(1)
        with torch.no_grad():
            ref_seq = torch.cat((previous, reference), dim=1)
            ref_r = (self.residual(ref_seq, vp) / scale)[:, :, m:-m, m:-m]
            diag = (ref_r.square().mean(dim=(-2, -1)) * center_valid).sum(dim=1)
            diag = diag / center_valid.sum(dim=1).clamp_min(1)
        return loss, diag
