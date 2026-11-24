from dataclasses import dataclass

import torch
from torch import nn

from neuralop.models import FNO


@dataclass
class FNOConfig:
    modes: int = 12
    width: int = 96
    layers: int = 4

    history: int = 5

    velocity_min: float = 1400.0
    velocity_max: float = 5000.0


class AutoregressiveFNO(nn.Module):
    """
    Thin task wrapper around the official NeuralOperator FNO.

    Raw FNO input channels:
      5 wavefield-history channels
      1 normalized velocity channel
      1 target-time channel
      1 normalized x-coordinate channel
      1 normalized z-coordinate channel

    Total: 9 channels

    The official FNO core itself is not modified.
    """

    def __init__(self, config=None):
        super().__init__()

        self.config = config or FNOConfig()

        if self.config.history != 5:
            raise ValueError(
                "This baseline expects history=5."
            )

        self.core = FNO(
            n_modes=(
                self.config.modes,
                self.config.modes,
            ),
            hidden_channels=self.config.width,
            in_channels=9,
            out_channels=1,
            n_layers=self.config.layers,
        )

        self.register_buffer(
            "_grid_cache",
            torch.empty(0),
            persistent=False,
        )

    def _grid(self, batch, nz, nx, device, dtype):
        z = torch.linspace(
            0.0,
            1.0,
            nz,
            device=device,
            dtype=dtype,
        )

        x = torch.linspace(
            0.0,
            1.0,
            nx,
            device=device,
            dtype=dtype,
        )

        zz, xx = torch.meshgrid(
            z,
            x,
            indexing="ij",
        )

        grid = torch.stack(
            (xx, zz),
            dim=0,
        )

        return grid.unsqueeze(0).expand(
            batch,
            -1,
            -1,
            -1,
        )

    def normalize_vp(self, vp):
        return (
            2.0
            * (
                vp
                - self.config.velocity_min
            )
            / (
                self.config.velocity_max
                - self.config.velocity_min
            )
            - 1.0
        )

    def encode_medium(self, vp):
        """
        Interface compatibility with the existing rollout trainers.

        No learned medium encoder is introduced here.
        """
        if vp.ndim == 3:
            vp = vp[:, None]

        return self.normalize_vp(vp)

    def predict_next(
        self,
        history,
        target_index,
        medium,
    ):
        if history.ndim != 4:
            raise ValueError(
                "history must be [B,5,Z,X]"
            )

        if history.shape[1] != 5:
            raise ValueError(
                f"Expected 5 history frames, "
                f"got {history.shape[1]}"
            )

        b, _, nz, nx = history.shape

        if medium.ndim == 3:
            medium = medium[:, None]

        if medium.shape != (
            b,
            1,
            nz,
            nx,
        ):
            raise ValueError(
                f"Unexpected medium shape: "
                f"{tuple(medium.shape)}"
            )

        # Same target-index information available
        # to all models, represented as a scalar field.
        t = (
            target_index
            .to(
                device=history.device,
                dtype=history.dtype,
            )
            .reshape(b, 1, 1, 1)
            / 100.0
        )

        t = t.expand(
            b,
            1,
            nz,
            nx,
        )

        grid = self._grid(
            b,
            nz,
            nx,
            history.device,
            history.dtype,
        )

        x = torch.cat(
            (
                history,
                medium,
                t,
                grid,
            ),
            dim=1,
        )

        return self.core(x)

    def forward(
        self,
        history,
        vp,
        target_index,
        horizon=1,
    ):
        medium = self.encode_medium(vp)

        predicted = []
        buffer = history

        for k in range(horizon):
            nxt = self.predict_next(
                buffer,
                target_index + k,
                medium,
            )

            predicted.append(nxt)

            buffer = torch.cat(
                (
                    nxt,
                    buffer[:, :-1],
                ),
                dim=1,
            )

        return torch.cat(
            predicted,
            dim=1,
        )
