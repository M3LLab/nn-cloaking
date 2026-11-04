"""Configuration for the fully embedded circular cloak (bulk P/S waves).

Everything is in SI units and absolute coordinates. The *physical box*
``[x_min, x_max] x [y_min, y_max]`` is surrounded on all four sides by an
absorbing layer of thickness ``absorbing.thickness``; the outer edge of the
layer is clamped (u = 0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, model_validator


class MaterialCfg(BaseModel):
    """Isotropic background (paper: lam = mu = 144 MPa, rho = 1600 kg/m^3)."""
    lam: float = 144e6
    mu: float = 144e6
    rho: float = 1600.0

    @property
    def cs(self) -> float:
        return float(np.sqrt(self.mu / self.rho))

    @property
    def cp(self) -> float:
        return float(np.sqrt((self.lam + 2 * self.mu) / self.rho))


class CloakCfg(BaseModel):
    xc: float = 0.0
    yc: float = 0.0
    a: float = 0.5            # cavity radius
    b: float = 1.5            # outer cloak radius (bonded interface)
    eps: float = 0.005        # virtual-hole radius of the regularised map
    radial_map: Literal["linear", "quadratic"] = "linear"

    @model_validator(mode="after")
    def _check(self):
        if not (0.0 <= self.eps < self.a < self.b):
            raise ValueError(f"need 0 <= eps < a < b; got {self.eps}, {self.a}, {self.b}")
        return self


class SourceCfg(BaseModel):
    """Gaussian body force f(x) = F0 d g(x) / int(g), g = exp(-|x-xs|^2 / 2 sigma^2).

    ``F0`` is the total force per unit thickness [N/m]. The normalising
    integral is evaluated with the *same* FEM quadrature used in the solve,
    so the discrete total force equals F0 on every mesh.
    """
    x: float = -3.0
    y: float = -3.0
    direction: tuple[float, float] = (0.0, 1.0)   # vertical polarisation
    amplitude: float = 1.0e3                       # F0 [N/m]
    sigma: float = 0.075                           # Gaussian width [m]
    frequency_hz: float = 200.0


class DomainCfg(BaseModel):
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -5.0
    y_max: float = 5.0


class AbsorbingCfg(BaseModel):
    """Complex-stiffness absorbing layer: C -> (1 + i xi) C, rho -> (1 - i xi) rho,
    xi(d) = xi_max (d / L)^power, summed over x and y in the corners."""
    thickness: float = 4.0
    xi_max: float = 4.0
    power: int = 2


class MeshCfg(BaseModel):
    """Target element sizes. Lengths derive from the background S wavelength.

    - ``elems_per_wavelength``: background size h_bg = lambda_S / n.
    - ``cloak_refine``: h in the annulus = h_bg / cloak_refine.
    - ``inner_bl_factor``: h at r = a = min(h_cloak, inner_bl_factor * delta),
      delta = eps * dr/dR(eps) is the physical thickness of the boundary layer
      that the virtual eps-hole's near field occupies next to the cavity.
      ``inner_h_max`` caps it (useful when eps = 0).
    - ``inner_grading``: distance from r = a over which h grows back to h_cloak.
    - ``source_h``: h at the source centre, as a fraction of sigma.
    - ``scale``: global multiplier on every size (mesh-convergence knob).
    """
    ele_type: Literal["TRI6", "TRI3"] = "TRI6"
    gauss_order: int = 4
    elems_per_wavelength: float = 8.0
    cloak_refine: float = 2.0
    inner_bl_factor: float = 2.0
    inner_h_max: float = 0.02
    inner_h_min: float = 1e-4
    inner_grading: float = 0.3
    source_h: float = 0.5
    scale: float = 1.0
    hole_h_factor: float = 0.5     # case D: h at the eps-hole = factor * eps


class MetricsCfg(BaseModel):
    grid_spacing: float = 0.05           # sampling grid spacing [m]
    exterior_margin: float = 0.02        # sample only r > b (1 + margin)
    source_exclusion: float = 0.75       # drop |x - xs| < this [m]
    box_margin: float = 0.0              # extra inset from the absorbing layer
    ring_radii_over_b: list[float] = [1.25, 2.0]
    ring_points: int = 720


class EmbeddedCloakConfig(BaseModel):
    material: MaterialCfg = MaterialCfg()
    cloak: CloakCfg = CloakCfg()
    source: SourceCfg = SourceCfg()
    domain: DomainCfg = DomainCfg()
    absorbing: AbsorbingCfg = AbsorbingCfg()
    mesh: MeshCfg = MeshCfg()
    metrics: MetricsCfg = MetricsCfg()
    output_dir: str = "output/circular_embedded"

    @model_validator(mode="after")
    def _check_layout(self):
        c, s, d = self.cloak, self.source, self.domain
        if not (d.x_min < c.xc - c.b and c.xc + c.b < d.x_max
                and d.y_min < c.yc - c.b and c.yc + c.b < d.y_max):
            raise ValueError("cloak must lie strictly inside the physical box")
        if np.hypot(s.x - c.xc, s.y - c.yc) < c.b + 4 * s.sigma:
            raise ValueError("source Gaussian overlaps the cloak (need |xs - c| > b + 4 sigma)")
        if not (d.x_min + 4 * s.sigma < s.x < d.x_max - 4 * s.sigma
                and d.y_min + 4 * s.sigma < s.y < d.y_max - 4 * s.sigma):
            raise ValueError("source Gaussian reaches into the absorbing layer")
        n = np.hypot(*s.direction)
        if n == 0:
            raise ValueError("source.direction must be non-zero")
        return self

    # derived quantities ------------------------------------------------
    @property
    def omega(self) -> float:
        return 2 * np.pi * self.source.frequency_hz

    @property
    def lambda_s(self) -> float:
        return self.material.cs / self.source.frequency_hz

    @property
    def lambda_p(self) -> float:
        return self.material.cp / self.source.frequency_hz

    @property
    def outer_box(self) -> tuple[float, float, float, float]:
        d, L = self.domain, self.absorbing.thickness
        return d.x_min - L, d.x_max + L, d.y_min - L, d.y_max + L

    def with_updates(self, updates: dict) -> "EmbeddedCloakConfig":
        """Return a copy with dotted-key overrides, e.g. {'cloak.eps': 0.01}."""
        data = self.model_dump()
        for key, val in updates.items():
            node = data
            parts = key.split(".")
            for p in parts[:-1]:
                node = node[p]
            if parts[-1] not in node:
                raise KeyError(f"unknown config key {key!r}")
            node[parts[-1]] = val
        return EmbeddedCloakConfig(**data)


def load_embedded_config(path: str | Path) -> EmbeddedCloakConfig:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data.pop("studies", None)   # study definitions are read by the driver script
    return EmbeddedCloakConfig(**data)
