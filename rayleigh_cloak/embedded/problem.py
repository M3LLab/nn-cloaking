"""Time-harmonic plane-strain elastodynamics for the embedded circular cloak.

Strong form (time factor exp(+i omega t), consistent with the damping sign):

    d/dx_j [ C^c_ijkl du_k/dx_l ] + omega^2 rho^c u_i + f_i = 0

* ``C^c``, ``rho^c`` are complex and evaluated at every quadrature point:
  background (C0, rho0) in the physical box, except that in case C the annulus
  elements receive the full BGM push-forward ``C' = F F C0 / J`` (no minor
  symmetry imposed) and ``rho' = rho0 / J``.
* Outside the box they carry the absorbing layer (``AbsorbingCfg``): by default
  a PML, ``C^c_ijkl = C_ijkl s_x s_y / (s_j s_l)``, ``rho^c = rho s_x s_y``.
* ``f`` is a Gaussian body force whose integral -- computed with the same
  quadrature as the solve -- equals ``F0 * direction`` exactly.

The cavity boundary carries no surface term (natural, traction-free BC); the
outer edge of the absorbing frame is clamped. The DOF layout per node is
[Re ux, Re uy, Im ux, Im uy] (jax-fem works in real arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
from rayleigh_cloak.embedded.mesh import REGION_ANNULUS, TaggedMesh
from rayleigh_cloak.geometry.circular import CircularCloakGeometry
from rayleigh_cloak.materials import C_iso, C_pushforward


@dataclass
class ProblemSpec:
    cfg: EmbeddedCloakConfig
    regions: np.ndarray        # (n_cells,)
    cloak_on: bool


def make_geometry(cfg: EmbeddedCloakConfig) -> CircularCloakGeometry:
    c = cfg.cloak
    return CircularCloakGeometry(ri=c.a, rc=c.b, x_c=c.xc, y_c=c.yc,
                                 eps=c.eps, radial_map=c.radial_map)


def pml_sigma_max(cfg: EmbeddedCloakConfig) -> float:
    ab = cfg.absorbing
    return (ab.power + 1) * cfg.material.cp * np.log(1.0 / ab.reflection) / (2.0 * ab.thickness)


def _layer_depths(cfg: EmbeddedCloakConfig, x):
    d = cfg.domain
    dx = jnp.maximum(jnp.maximum(d.x_min - x[0], x[0] - d.x_max), 0.0)
    dy = jnp.maximum(jnp.maximum(d.y_min - x[1], x[1] - d.y_max), 0.0)
    return dx, dy


def layer_loss_profile(cfg: EmbeddedCloakConfig):
    """Scalar loss map for plots: (sigma_x + sigma_y)/omega (pml) or xi (damping)."""
    ab = cfg.absorbing
    amp = pml_sigma_max(cfg) / cfg.omega if ab.kind == "pml" else ab.xi_max

    def loss(x):
        dx, dy = _layer_depths(cfg, x)
        return amp * ((dx / ab.thickness) ** ab.power + (dy / ab.thickness) ** ab.power)

    return loss


def layer_coefficients(cfg: EmbeddedCloakConfig):
    """Return ``fn(x, C, rho) -> (C^c, rho^c)`` applying the absorbing layer."""
    ab = cfg.absorbing
    L, p = ab.thickness, ab.power
    if ab.kind == "damping":
        def damp(x, C, rho):
            dx, dy = _layer_depths(cfg, x)
            xi = ab.xi_max * ((dx / L) ** p + (dy / L) ** p)
            return (1.0 + 1j * xi) * C, (1.0 - 1j * xi) * rho
        return damp

    amp = pml_sigma_max(cfg) / cfg.omega

    def pml(x, C, rho):
        dx, dy = _layer_depths(cfg, x)
        s = jnp.stack([1.0 - 1j * amp * (dx / L) ** p, 1.0 - 1j * amp * (dy / L) ** p])
        J = s[0] * s[1]
        # the derivative sits on indices j and l of C_ijkl
        scale = J / (s[None, :, None, None] * s[None, None, None, :])
        return C * scale, rho * J
    return pml


class EmbeddedCloakProblem(Problem):

    def custom_init(self, spec: ProblemSpec):
        cfg = spec.cfg
        mat, src = cfg.material, cfg.source
        self._omega = cfg.omega
        qp = jnp.asarray(self.physical_quad_points)            # (nc, nq, 2)
        nc, nq, _ = qp.shape

        C0 = C_iso(mat.lam, mat.mu)
        C_qp = jnp.broadcast_to(C0, (nc, nq, 2, 2, 2, 2))
        rho_qp = jnp.full((nc, nq), mat.rho)
        if spec.cloak_on:
            geo = make_geometry(cfg)
            sel = np.nonzero(np.asarray(spec.regions) == REGION_ANNULUS)[0]
            xq = qp[sel].reshape(-1, 2)
            F = jax.vmap(lambda x: geo.F_cloak(x, clamp=True))(xq)
            Cc = jax.vmap(lambda Fi: C_pushforward(Fi, C0))(F)
            Jc = jnp.linalg.det(F)
            C_qp = C_qp.at[sel].set(Cc.reshape(len(sel), nq, 2, 2, 2, 2))
            rho_qp = rho_qp.at[sel].set((mat.rho / Jc).reshape(len(sel), nq))
            self.cloak_stats = dict(J_min=float(Jc.min()), J_max=float(Jc.max()),
                                    C_max_over_mu=float(jnp.abs(Cc).max() / mat.mu))

        layer = layer_coefficients(cfg)
        Cc_qp, rhoc_qp = jax.vmap(jax.vmap(layer))(qp, C_qp, rho_qp)

        # Gaussian body force, normalised with the solve's own quadrature
        g = jnp.exp(-0.5 * ((qp[..., 0] - src.x) ** 2 + (qp[..., 1] - src.y) ** 2)
                    / src.sigma ** 2)
        JxW = jnp.asarray(self.fes[0].JxW)                       # (nc, nq)
        g_int = float(jnp.sum(g * JxW))
        dvec = np.asarray(src.direction, dtype=float)
        dvec = dvec / np.linalg.norm(dvec)
        f_qp = (src.amplitude / g_int) * g[..., None] * jnp.asarray(dvec)[None, None, :]
        self.source_stats = dict(
            discrete_integral=g_int,
            analytic_integral=float(2 * np.pi * src.sigma ** 2),
            ratio=g_int / float(2 * np.pi * src.sigma ** 2),
            total_force=[float(jnp.sum(f_qp[..., k] * JxW)) for k in range(2)],
        )
        self.internal_vars = [Cc_qp.real, Cc_qp.imag, rhoc_qp.real, rhoc_qp.imag, f_qp]

    def get_tensor_map(self):
        def stress(u_grad, CR, CI, _rR, _rI, _f):
            gR, gI = u_grad[:2, :], u_grad[2:, :]
            sR = jnp.einsum("ijkl,kl->ij", CR, gR) - jnp.einsum("ijkl,kl->ij", CI, gI)
            sI = jnp.einsum("ijkl,kl->ij", CR, gI) + jnp.einsum("ijkl,kl->ij", CI, gR)
            return jnp.concatenate([sR, sI], axis=0)
        return stress

    def get_mass_map(self):
        omega = self._omega

        def inertia(u, _x, _CR, _CI, rR, rI, f):
            uR, uI = u[:2], u[2:]
            mR = -omega ** 2 * (rR * uR - rI * uI) - f
            mI = -omega ** 2 * (rR * uI + rI * uR)
            return jnp.concatenate([mR, mI])
        return inertia


def clamped_outer_bc(cfg: EmbeddedCloakConfig):
    X0, X1, Y0, Y1 = cfg.outer_box
    tol = 1e-8 * max(X1 - X0, Y1 - Y0)

    def on_outer(p):
        return ((jnp.abs(p[0] - X0) < tol) | (jnp.abs(p[0] - X1) < tol)
                | (jnp.abs(p[1] - Y0) < tol) | (jnp.abs(p[1] - Y1) < tol))

    zero = lambda p: 0.0
    return [[on_outer] * 4, [0, 1, 2, 3], [zero] * 4]


def build_problem(mesh: TaggedMesh, cfg: EmbeddedCloakConfig, cloak_on: bool):
    spec = ProblemSpec(cfg=cfg, regions=mesh.region, cloak_on=cloak_on)
    return EmbeddedCloakProblem(
        mesh=mesh.to_jaxfem(), vec=4, dim=2, ele_type=mesh.ele_type,
        gauss_order=cfg.mesh.gauss_order,
        dirichlet_bc_info=clamped_outer_bc(cfg),
        additional_info=(spec,),
    )
