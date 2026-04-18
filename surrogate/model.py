"""
NN model to predict transmission loss from cell geometry.

Shared CNN trunk → latent geometry code z; continuous Fourier-features
decoder maps (z, f) → scalar loss. Two call modes share all weights:
    forward_at       — one f per sample, returns (B,)
    forward_spectrum — F-point grid broadcast across the batch, returns (B, F)
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torchvision.models import convnext_small

from surrogate.dataset import SurrogateBatch


class FrequencyOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    predicted_cloaking: Float[Tensor, "B"]


class SpectrumOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    # B - batch size
    # F - number of frequencies
    predicted_cloaking: Float[Tensor, "B F"]


class FourierFeatures(nn.Module):
    """Sinusoidal encoding of a scalar frequency.

    Normalizes f to [0, 1] over [f_min, f_max], then applies
    [sin(π k f̂), cos(π k f̂)] for k = 1..num_bands.
    """

    def __init__(self, num_bands: int = 8, f_min: float = 1.0, f_max: float = 4.0):
        super().__init__()
        self.f_min = f_min
        self.f_max = f_max
        bands = torch.arange(1, num_bands + 1, dtype=torch.float32) * math.pi
        self.register_buffer("bands", bands)
        self.out_dim = 2 * num_bands

    def forward(self, f: Tensor) -> Tensor:
        f_norm = (f - self.f_min) / (self.f_max - self.f_min)
        angles = f_norm.unsqueeze(-1) * self.bands
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class SpectrumDecoder(nn.Module):
    """MLP that maps concat(z, Fourier(f)) → scalar prediction."""

    def __init__(self, z_dim: int, ff_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + ff_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),          nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_ff: Tensor) -> Tensor:
        return self.net(z_ff).squeeze(-1)


class ForwardFEM_CNN(nn.Module):
    def __init__(
        self,
        n_C_params: int,
        z_dim: int = 128,
        fourier_bands: int = 8,
        f_min: float = 1.0,
        f_max: float = 4.0,
        decoder_hidden: int = 128,
    ):
        super().__init__()
        in_channels = n_C_params + 1  # C params stacked with rho

        self.trunk = convnext_small()
        stem_conv: nn.Conv2d = self.trunk.features[0][0]
        self.trunk.features[0][0] = nn.Conv2d(
            in_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
        )
        final_linear: nn.Linear = self.trunk.classifier[-1]
        self.trunk.classifier[-1] = nn.Linear(final_linear.in_features, z_dim)

        self.fourier = FourierFeatures(fourier_bands, f_min, f_max)
        self.decoder = SpectrumDecoder(z_dim, self.fourier.out_dim, decoder_hidden)

    def encode(self, batch: SurrogateBatch) -> Tensor:
        """Run the trunk once; returns the geometry latent code z, shape (B, d)."""
        x = torch.cat([batch.C, batch.rho.unsqueeze(-1)], dim=-1)  # B X Y (P+1)
        x = rearrange(x, "b x y c -> b c x y")
        return self.trunk(x)

    def forward_at(
        self,
        batch: SurrogateBatch,
        f: Tensor | None = None,
    ) -> FrequencyOutput:
        """One frequency per sample (shape B). Defaults to batch.f_star."""
        if f is None:
            f = batch.f_star
        z = self.encode(batch)
        ff = self.fourier(f)
        pred = self.decoder(torch.cat([z, ff], dim=-1))
        return FrequencyOutput(predicted_cloaking=pred)

    def forward_spectrum(
        self,
        batch: SurrogateBatch,
        f_grid: Tensor,
    ) -> SpectrumOutput:
        """Shared frequency grid (shape F) evaluated for every sample."""
        z = self.encode(batch)                                # (B, d)
        ff = self.fourier(f_grid)                             # (F, FF)
        n_batch, n_freq = z.shape[0], f_grid.shape[0]
        z_rep = z.unsqueeze(1).expand(-1, n_freq, -1)         # (B, F, d)
        ff_rep = ff.unsqueeze(0).expand(n_batch, -1, -1)      # (B, F, FF)
        pred = self.decoder(torch.cat([z_rep, ff_rep], dim=-1))
        return SpectrumOutput(predicted_cloaking=pred)

    def forward(self, batch: SurrogateBatch) -> FrequencyOutput:
        return self.forward_at(batch)

