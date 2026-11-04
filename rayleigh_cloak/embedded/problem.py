"""Time-harmonic plane-strain elastodynamics for the embedded circular cloak.

Strong form (time factor exp(+i omega t), consistent with the damping sign):

    d/dx_j [ (1 + i xi) C_ijkl du_k/dx_l ] + omega^2 (1 - i xi) rho u_i + f_i = 0

* ``C``, ``rho`` are evaluated at every quadrature point: background (C0, rho0)
  everywhere except, in case C, the annulus elements, which receive the full
  BGM push-forward ``C' = F F C0 / J`` (no minor symmetry imposed) and
  ``rho' = rho0 / J``.
* ``xi`` is the absorbing-layer profile (zero in the physical box).
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


def xi_profile(cfg: EmbeddedCloakConfig):
    d, ab = cfg.domain, cfg.absorbing
    L, xm, p = ab.thickness, ab.xi_max, ab.power

    def xi(x):
        dx = jnp.maximum(jnp.maximum(d.x_min - x[0], x[0] - d.x_max), 0.0)
        dy = jnp.maximum(jnp.maximum(d.y_min - x[1], x[1] - d.y_max), 0.0)
        return xm * ((dx / L) ** p + (dy / L) ** p)

    return xi


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

        xi_qp = jax.vmap(jax.vmap(xi_profile(cfg)))(qp)

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
        self.internal_vars = [C_qp, rho_qp, xi_qp, f_qp]

    def get_tensor_map(self):
        def stress(u_grad, C, _rho, xi, _f):
            gR, gI = u_grad[:2, :], u_grad[2:, :]
            sR = jnp.einsum("ijkl,kl->ij", C, gR)
            sI = jnp.einsum("ijkl,kl->ij", C, gI)
            return jnp.concatenate([sR - xi * sI, sI + xi * sR], axis=0)
        return stress

    def get_mass_map(self):
        omega = self._omega

        def inertia(u, _x, _C, rho, xi, f):
            uR, uI = u[:2], u[2:]
            mR = -omega ** 2 * rho * (uR + xi * uI) - f
            mI = -omega ** 2 * rho * (uI - xi * uR)
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
