"""Pydantic configuration models and derived physical parameters.

All user-facing parameters live in ``SimulationConfig`` (serialisable to/from
YAML).  Computed quantities that depend on multiple config fields are collected
in ``DerivedParams`` so they are calculated once and threaded through the code
without module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel


# ── user-facing config sections ──────────────────────────────────────


class MaterialConfig(BaseModel):
    rho0: float = 1600.0
    cs: float = 300.0


class DomainConfig(BaseModel):
    f_star: float = 2.0
    lambda_star: float = 1.0
    H_factor: float = 4.305  # H = H_factor * lambda_star
    W_factor: float = 12.5   # W = W_factor * lambda_star


class TriangularCloakConfig(BaseModel):
    a_factor: float = 0.0774   # inner depth  = a_factor * H
    b_factor: float = 3.0      # outer depth  = b_factor * a
    c_factor: float = 0.1545   # half-width   = c_factor * H


class AbsorbingConfig(BaseModel):
    L_pml_factor: float = 1.0  # L_pml = factor * lambda_star
    xi_max: float = 4.0
    pml_pow: int = 2


class MeshConfig(BaseModel):
    n_pml_x: int = 32
    n_pml_y: int = 32
    nx_phys: int = 200
    ny_phys: int = 60
    refinement_factor: int = 4
    ele_type: str = "TRI3"


class SourceConfig(BaseModel):
    x_src_factor: float = 0.05   # x_src = factor * W
    sigma_factor: float = 0.01   # sigma = factor * lambda_star
    F0: float = 1.0


class SolverConfig(BaseModel):
    ksp_type: str = "preonly"
    pc_type: str = "lu"


class SimulationConfig(BaseModel):
    is_reference: bool = False
    geometry_type: str = "triangular"

    material: MaterialConfig = MaterialConfig()
    domain: DomainConfig = DomainConfig()
    geometry: TriangularCloakConfig = TriangularCloakConfig()
    absorbing: AbsorbingConfig = AbsorbingConfig()
    mesh: MeshConfig = MeshConfig()
    source: SourceConfig = SourceConfig()
    solver: SolverConfig = SolverConfig()
    output_dir: str = "output"


def load_config(path: str | Path) -> SimulationConfig:
    """Load a ``SimulationConfig`` from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimulationConfig(**(data or {}))


# ── derived / computed quantities ────────────────────────────────────


@dataclass(frozen=True)
class DerivedParams:
    """Physical quantities derived from :class:`SimulationConfig`.

    All lengths are in the *extended-mesh* coordinate system (i.e. including
    PML padding).
    """

    # material
    rho0: float
    mu: float
    lam: float
    nu: float
    cs: float
    cp: float
    cR: float

    # frequency
    omega: float

    # domain extents (physical)
    H: float
    W: float
    lambda_star: float

    # PML
    L_pml: float
    xi_max: float
    pml_pow: int

    # extended domain
    W_total: float
    H_total: float
    x_off: float
    y_off: float

    # convenience
    y_top: float
    x_c: float  # cloak centre x

    # cloak geometry
    a: float
    b: float
    c: float

    # source
    x_src: float
    x_src_phys: float
    sigma_src: float
    F0: float

    # mesh
    nx_total: int
    ny_total: int

    @staticmethod
    def from_config(cfg: SimulationConfig) -> DerivedParams:
        mat = cfg.material
        dom = cfg.domain
        geo = cfg.geometry
        ab = cfg.absorbing
        msh = cfg.mesh
        src = cfg.source

        rho0 = mat.rho0
        cs = mat.cs
        cp = np.sqrt(3.0) * cs
        mu = rho0 * cs ** 2
        lam = rho0 * cp ** 2 - 2 * mu
        nu = lam / (2 * (lam + mu))
        cR = cs * (0.826 + 1.14 * nu) / (1 + nu)

        lambda_star = dom.lambda_star
        omega = 2 * np.pi * dom.f_star * cR / lambda_star

        H = dom.H_factor * lambda_star
        W = dom.W_factor * lambda_star

        a = geo.a_factor * H
        b = geo.b_factor * a
        c = geo.c_factor * H

        L_pml = ab.L_pml_factor * lambda_star
        W_total = 2 * L_pml + W
        H_total = L_pml + H
        x_off = L_pml
        y_off = L_pml
        y_top = H_total
        x_c = x_off + W / 2.0

        x_src_phys = src.x_src_factor * W
        x_src = x_off + x_src_phys
        sigma_src = src.sigma_factor * lambda_star

        nx_total = msh.n_pml_x + msh.nx_phys + msh.n_pml_x
        ny_total = msh.n_pml_y + msh.ny_phys

        return DerivedParams(
            rho0=rho0, mu=mu, lam=lam, nu=nu, cs=cs, cp=cp, cR=cR,
            omega=omega,
            H=H, W=W, lambda_star=lambda_star,
            L_pml=L_pml, xi_max=ab.xi_max, pml_pow=ab.pml_pow,
            W_total=W_total, H_total=H_total, x_off=x_off, y_off=y_off,
            y_top=y_top, x_c=x_c,
            a=a, b=b, c=c,
            x_src=x_src, x_src_phys=x_src_phys,
            sigma_src=sigma_src, F0=src.F0,
            nx_total=nx_total, ny_total=ny_total,
        )
