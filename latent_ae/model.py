"""Frequency-aware autoencoder.

Architecture:
    Encoder  : SmallConvNeXt(in=n_C_params+1, out=2*z_dim) → (z_sh, z_f_shift)
    FreqEnc  : MLP(Fourier(f))                             → z_f(f)
    Latent   : z = z_sh + z_f(f) + z_f_shift
    Decoder  : Linear + reshape + ConvNeXt up-stages       → (C, rho)
    PerfHead : MLP(z)                                      → scalar loss

Loss breakdown lives in ``latent_ae/loss.py``. The ranking loss acts only on
``z_sh`` — that's the frequency-independent branch, where "good designs at
any frequency" should cluster.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from surrogate.dataset import SurrogateBatch
from surrogate.model import (
    ConvNeXtBlock,
    FourierFeatures,
    LayerNorm2d,
    SmallConvNeXt,
)


class EncodeOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    z_sh:        Float[Tensor, "B D"]
    z_f:         Float[Tensor, "B D"]
    z_f_shift:   Float[Tensor, "B D"]
    z:           Float[Tensor, "B D"]
    C_recon:     Float[Tensor, "B X Y P"]
    rho_recon:   Float[Tensor, "B X Y"]
    loss_pred:   Float[Tensor, "B"]


class FreqEncoder(nn.Module):
    """Fourier(f) → z_dim residual."""

    def __init__(self, ff_dim: int, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ff_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, ff: Tensor) -> Tensor:
        return self.net(ff)


class PerfHead(nn.Module):
    """z → predicted (transformed) loss scalar."""

    def __init__(self, z_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


class ConvNeXtDecoder(nn.Module):
    """Mirror of SmallConvNeXt. z → (out_channels, X, Y).

    Projects z to a coarse (dims[0], H0, W0) grid, then repeated
    nearest-upsample + Conv + ConvNeXt blocks to reach target (X, Y).
    """

    def __init__(
        self,
        z_dim: int,
        out_channels: int,
        target_hw: tuple[int, int],
        dims: tuple[int, int, int] = (256, 128, 64),
        depths: tuple[int, int, int] = (2, 2, 2),
        kernel: int = 7,
    ):
        super().__init__()
        self.target_hw = target_hw
        # Start at a coarse grid — ceil(target / 4). The SmallConvNeXt encoder
        # downsamples ×4 total (two stride-2 stages), so we mirror that here.
        h0 = max(1, (target_hw[0] + 3) // 4)
        w0 = max(1, (target_hw[1] + 3) // 4)
        self.h0, self.w0 = h0, w0

        self.project = nn.Linear(z_dim, dims[0] * h0 * w0)
        self.project_norm = LayerNorm2d(dims[0])

        self.stages = nn.ModuleList([
            nn.Sequential(*[ConvNeXtBlock(dims[0], kernel) for _ in range(depths[0])])
        ])
        self.upsamples = nn.ModuleList()
        for i in range(1, len(dims)):
            # nearest-upsample + pointwise conv to change channels
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                LayerNorm2d(dims[i - 1]),
                nn.Conv2d(dims[i - 1], dims[i], kernel_size=1),
            ))
            self.stages.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i], kernel) for _ in range(depths[i])]
            ))

        self.head_norm = LayerNorm2d(dims[-1])
        self.head = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

    def forward(self, z: Tensor) -> Tensor:
        B = z.shape[0]
        x = self.project(z).view(B, -1, self.h0, self.w0)
        x = self.project_norm(x)
        x = self.stages[0](x)
        for up, stage in zip(self.upsamples, self.stages[1:]):
            x = stage(up(x))
        # Crop / pad to exact target (X, Y)
        h, w = x.shape[-2], x.shape[-1]
        tx, ty = self.target_hw
        if h != tx or w != ty:
            x = F.interpolate(x, size=(tx, ty), mode="bilinear", align_corners=False)
        x = self.head_norm(x)
        return self.head(x)


class LatentAutoencoder(nn.Module):
    def __init__(
        self,
        n_C_params: int,
        grid_hw: tuple[int, int],
        cloak_mask: Tensor | None = None,
        z_dim: int = 128,
        fourier_bands: int = 16,
        f_min: float = 0.1,
        f_max: float = 4.0,
        residual_hidden: int = 128,
        decoder_dims: tuple[int, int, int] = (256, 128, 64),
        perf_hidden: int = 128,
    ):
        super().__init__()
        self.n_C_params = n_C_params
        self.grid_hw = grid_hw
        self.z_dim = z_dim

        in_channels = n_C_params + 1  # C stacked with rho
        # Trunk produces 2*z_dim: [z_sh | z_f_shift]
        self.trunk = SmallConvNeXt(in_channels=in_channels, out_dim=2 * z_dim)

        self.fourier = FourierFeatures(fourier_bands, f_min, f_max)
        self.freq_encoder = FreqEncoder(self.fourier.out_dim, z_dim, residual_hidden)

        self.decoder = ConvNeXtDecoder(
            z_dim=z_dim,
            out_channels=in_channels,
            target_hw=grid_hw,
            dims=decoder_dims,
        )

        self.perf_head = PerfHead(z_dim, perf_hidden)

        if cloak_mask is not None:
            # (X, Y) bool — register as buffer so it moves with .to(device)
            self.register_buffer("cloak_mask", cloak_mask.to(torch.float32))
        else:
            self.cloak_mask = None

    # ── Encoder helpers ─────────────────────────────────────────────

    def _stack_input(self, batch: SurrogateBatch) -> Tensor:
        x = torch.cat([batch.C, batch.rho.unsqueeze(-1)], dim=-1)  # B X Y (P+1)
        return rearrange(x, "b x y c -> b c x y")

    def encode_raw(self, batch: SurrogateBatch) -> tuple[Tensor, Tensor]:
        """Run the trunk and split into (z_sh, z_f_shift)."""
        h = self.trunk(self._stack_input(batch))         # (B, 2*z_dim)
        z_sh, z_f_shift = h.chunk(2, dim=-1)
        return z_sh, z_f_shift

    def encode_shared(self, batch: SurrogateBatch) -> Tensor:
        """Frequency-independent branch only (used at optimization init)."""
        z_sh, _ = self.encode_raw(batch)
        return z_sh

    def freq_residual(self, f: Tensor) -> Tensor:
        return self.freq_encoder(self.fourier(f))

    # ── Full forward ────────────────────────────────────────────────

    def encode(self, batch: SurrogateBatch) -> EncodeOutput:
        z_sh, z_f_shift = self.encode_raw(batch)
        z_f = self.freq_residual(batch.f_star)
        z = z_sh + z_f + z_f_shift
        C_recon, rho_recon = self.decode(z)
        loss_pred = self.perf_head(z)
        return EncodeOutput(
            z_sh=z_sh, z_f=z_f, z_f_shift=z_f_shift, z=z,
            C_recon=C_recon, rho_recon=rho_recon, loss_pred=loss_pred,
        )

    def decode(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """z → (C, rho). Applies cloak mask (hard-zero outside cloak) when set."""
        out = self.decoder(z)                            # (B, P+1, X, Y)
        out = rearrange(out, "b c x y -> b x y c")
        C, rho = out[..., : self.n_C_params], out[..., self.n_C_params]
        if self.cloak_mask is not None:
            C = C * self.cloak_mask[None, :, :, None]
            rho = rho * self.cloak_mask[None, :, :]
        return C, rho

    def predict_loss(self, z: Tensor) -> Tensor:
        return self.perf_head(z)

    def forward(self, batch: SurrogateBatch) -> EncodeOutput:
        return self.encode(batch)
