"""Material pixels in the cloak annulus, independent of the FEM mesh.

A polar grid of ``n_theta x n_r`` pixels, uniform in theta and r, covers a < r < b.
Each pixel holds one stiffness tensor and one density. The pixel of a quadrature point is
found from its coordinates, so the pixel grid does not follow the mesh, and the cavity and
r = b stay exact because they are mesh edges. Case A/B background values are untouched
outside the annulus.

``sample_ideal`` computes pixel values by integrating the ideal material on a sub-grid inside
each pixel (or at its centre), so that they do not depend on the FEM mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
from rayleigh_cloak.embedded.mesh import REGION_ANNULUS, TaggedMesh
from rayleigh_cloak.embedded.problem import build_problem, make_geometry
from rayleigh_cloak.materials import C_iso, C_pushforward, symmetrize_stiffness


@dataclass(frozen=True)
class PolarPixelGrid:
    a: float
    b: float
    xc: float
    yc: float
    n_theta: int
    n_r: int

    @classmethod
    def from_config(cls, cfg: EmbeddedCloakConfig, n_theta: int, n_r: int) -> "PolarPixelGrid":
        c = cfg.cloak
        return cls(c.a, c.b, c.xc, c.yc, int(n_theta), int(n_r))

    @property
    def n_pixels(self) -> int:
        return self.n_theta * self.n_r

    def index(self, xy: np.ndarray) -> np.ndarray:
        """Pixel index of each point; radius is clipped to [a, b] for boundary quadrature points."""
        xy = np.asarray(xy)
        dx, dy = xy[..., 0] - self.xc, xy[..., 1] - self.yc
        r = np.clip(np.hypot(dx, dy), self.a, self.b)
        t = np.mod(np.arctan2(dy, dx), 2 * np.pi)
        ir = np.minimum(((r - self.a) / (self.b - self.a) * self.n_r).astype(np.int64), self.n_r - 1)
        it = np.minimum((t / (2 * np.pi) * self.n_theta).astype(np.int64), self.n_theta - 1)
        return it * self.n_r + ir

    def centres_polar(self) -> tuple[np.ndarray, np.ndarray]:
        """(r, theta) of the pixel centres, in index order."""
        dr = (self.b - self.a) / self.n_r
        dt = 2 * np.pi / self.n_theta
        it, ir = np.meshgrid(np.arange(self.n_theta), np.arange(self.n_r), indexing="ij")
        return (self.a + (ir.ravel() + 0.5) * dr, (it.ravel() + 0.5) * dt)

    def sample_points(self, sub: int) -> tuple[np.ndarray, np.ndarray]:
        """Sub-grid points and area weights, shape (n_pixels, sub^2, ...)."""
        dr = (self.b - self.a) / self.n_r
        dt = 2 * np.pi / self.n_theta
        rc, tc = self.centres_polar()
        off = (np.arange(sub) + 0.5) / sub - 0.5
        orr, ott = np.meshgrid(off, off, indexing="ij")
        r = rc[:, None] + dr * orr.ravel()[None, :]
        t = tc[:, None] + dt * ott.ravel()[None, :]
        pts = np.stack([self.xc + r * np.cos(t), self.yc + r * np.sin(t)], axis=-1)
        return pts, r


def ideal_at(cfg: EmbeddedCloakConfig, pts: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Ideal stiffness and density at points inside the annulus, shape (..., 2,2,2,2) and (...)."""
    m = cfg.material
    geo = make_geometry(cfg)
    C0 = C_iso(m.lam, m.mu)
    flat = jnp.asarray(pts).reshape(-1, 2)
    F = jax.vmap(lambda x: geo.F_cloak(x, clamp=True))(flat)
    C = jax.vmap(lambda Fi: C_pushforward(Fi, C0))(F)
    rho = m.rho / jnp.linalg.det(F)
    shape = pts.shape[:-1]
    return C.reshape(*shape, 2, 2, 2, 2), rho.reshape(shape)


def sample_ideal(cfg, grid: PolarPixelGrid, method: str = "average", sub: int = 8):
    """Pixel values (C, rho) of the ideal cloak. ``method``: 'average' or 'centre'.

    'average' is the area-weighted mean of C' and rho' over the pixel, on a ``sub x sub`` grid.
    """
    if method == "centre":
        r, t = grid.centres_polar()
        pts = np.stack([grid.xc + r * np.cos(t), grid.yc + r * np.sin(t)], axis=-1)
        return ideal_at(cfg, pts)
    if method != "average":
        raise ValueError(f"unknown sampling method {method!r}")
    pts, w = grid.sample_points(sub)
    C, rho = ideal_at(cfg, pts)
    w = jnp.asarray(w / w.sum(axis=1, keepdims=True))
    return jnp.einsum("ps,psijkl->pijkl", w, C), jnp.einsum("ps,ps->p", w, rho)


symmetrize_many = jax.vmap(symmetrize_stiffness)


class PixelProblem:
    """A jax-fem problem whose annulus material is set per pixel (or per quadrature point)."""

    def __init__(self, mesh: TaggedMesh, cfg: EmbeddedCloakConfig, grid: PolarPixelGrid | None = None):
        self.mesh, self.cfg, self.grid = mesh, cfg, grid
        self.problem = build_problem(mesh, cfg, cloak_on=False)
        self.sel = np.nonzero(np.asarray(mesh.region) == REGION_ANNULUS)[0]
        qp = np.asarray(self.problem.physical_quad_points)
        self.nq = qp.shape[1]
        self.xq = qp[self.sel]
        self.base = list(self.problem.internal_vars)
        self.qp_pixel = grid.index(self.xq) if grid is not None else None

    def active_pixels(self) -> int:
        """Pixels that contain at least one quadrature point on this mesh."""
        return int(np.unique(self.qp_pixel).size)

    def set_quadrature_material(self, C_q, rho_q) -> None:
        """Real annulus material at quadrature points, shape (n_sel, nq, 2,2,2,2) and (n_sel, nq)."""
        iv = list(self.base)
        iv[0] = iv[0].at[self.sel].set(C_q)
        iv[2] = iv[2].at[self.sel].set(rho_q)
        self.problem.internal_vars = iv

    def set_pixels(self, C_pix, rho_pix) -> None:
        idx = jnp.asarray(self.qp_pixel)
        self.set_quadrature_material(jnp.asarray(C_pix)[idx], jnp.asarray(rho_pix)[idx])

    def set_ideal(self, symmetrize: bool = False) -> None:
        C, rho = ideal_at(self.cfg, self.xq)
        if symmetrize:
            C = symmetrize_many(C.reshape(-1, 2, 2, 2, 2)).reshape(C.shape)
        self.set_quadrature_material(C, rho)
