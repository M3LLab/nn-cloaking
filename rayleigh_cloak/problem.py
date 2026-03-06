"""FEM problem definition for frequency-domain elastodynamics.

DOF ordering per node: [Re(ux), Re(uy), Im(ux), Im(uy)]  (vec = 4).

The ``RayleighCloakProblem`` class hooks into JAX-FEM via:
  - ``custom_init``   : precompute C_eff, rho_eff, xi at quadrature points
  - ``get_tensor_map`` : stiffness + stiffness-proportional damping coupling
  - ``get_mass_map``   : inertia  + mass-proportional damping coupling
  - ``get_surface_maps`` : Rayleigh-wave point source (real part only)
"""

from __future__ import annotations
from typing import Callable

import jax
import jax.numpy as jnp
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem

from rayleigh_cloak.absorbing import make_xi_profile
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry
from rayleigh_cloak.materials import C_eff, C_iso, rho_eff


class RayleighCloakProblem(Problem):
    """Frequency-domain elastodynamics with Rayleigh-damping absorbing layers.

    Additional attributes set before ``__init__``:
        _omega, _geometry, _C0, _rho0, _xi_fn, _x_src, _sigma_src, _F0
    """

    def custom_init(self):
        geo = self._geometry
        C0 = self._C0
        rho0 = self._rho0
        xi_fn = type(self).__dict__['_xi_fn']

        def _C_eff_pt(x):
            return C_eff(x, geo, C0)

        def _rho_eff_pt(x):
            return rho_eff(x, geo, rho0)

        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points),
        ]

    def get_tensor_map(self):
        def stress(u_grad, C_q, _rho_q, xi_q):
            grad_R = u_grad[:2, :]
            grad_I = u_grad[2:, :]

            sig_R_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_R)
            sig_I_undamped = jnp.einsum("ijkl,kl->ij", C_q, grad_I)

            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)

        return stress

    def get_mass_map(self):
        omega = self._omega

        def inertia(u, _x, _C_q, rho_q, xi_q):
            u_R, u_I = u[:2], u[2:]
            m_R = -omega ** 2 * rho_q * (u_R + xi_q * u_I)
            m_I = -omega ** 2 * rho_q * (u_I - xi_q * u_R)
            return jnp.concatenate([m_R, m_I])

        return inertia

    def get_surface_maps(self):
        x_src = self._x_src
        sigma_src = self._sigma_src
        F0 = self._F0

        def traction(_u, x):
            g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
            return jnp.array([0.0, g, 0.0, 0.0])

        return [traction]


# ── boundary conditions ──────────────────────────────────────────────


def _make_dirichlet_bc(params: DerivedParams):
    """Build the ``dirichlet_bc_info`` list for zero-displacement PML edges."""
    W_total = params.W_total

    def bc_bottom(point):
        return jnp.isclose(point[1], 0.0)

    def bc_left(point):
        return jnp.isclose(point[0], 0.0)

    def bc_right(point):
        return jnp.isclose(point[0], W_total)

    def zero(point):
        return 0.0

    return [
        [bc_bottom, bc_bottom, bc_bottom, bc_bottom,
         bc_left,   bc_left,   bc_left,   bc_left,
         bc_right,  bc_right,  bc_right,  bc_right],
        [0, 1, 2, 3,
         0, 1, 2, 3,
         0, 1, 2, 3],
        [zero, zero, zero, zero,
         zero, zero, zero, zero,
         zero, zero, zero, zero],
    ]


def _make_top_surface(params: DerivedParams) -> Callable:
    H_total = params.H_total

    def top_surface(point):
        return jnp.isclose(point[1], H_total)

    return top_surface


# ── factory ──────────────────────────────────────────────────────────


def build_problem(
    mesh: Mesh,
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> RayleighCloakProblem:
    """Assemble a ``RayleighCloakProblem`` ready for solving."""
    C0 = C_iso(params.lam, params.mu)

    # Inject dependencies via instance attributes *before* Problem.__init__
    # calls custom_init.
    RayleighCloakProblem._omega = params.omega
    RayleighCloakProblem._geometry = geometry
    RayleighCloakProblem._C0 = C0
    RayleighCloakProblem._rho0 = params.rho0
    RayleighCloakProblem._xi_fn = make_xi_profile(params)
    RayleighCloakProblem._x_src = params.x_src
    RayleighCloakProblem._sigma_src = params.sigma_src
    RayleighCloakProblem._F0 = params.F0

    problem = RayleighCloakProblem(
        mesh=mesh,
        vec=4,
        dim=2,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )
    return problem
