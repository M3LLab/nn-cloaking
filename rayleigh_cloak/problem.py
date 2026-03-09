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
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry
from rayleigh_cloak.materials import C_eff, C_iso, rho_eff, _get_converters


class RayleighCloakProblem(Problem):
    """Frequency-domain elastodynamics with Rayleigh-damping absorbing layers.

    Additional attributes set before ``__init__``:
        _omega, _geometry, _C0, _rho0, _xi_fn, _x_src, _sigma_src, _F0
    """

    def custom_init(self):
        geo = self._geometry
        C0 = self._C0
        rho0 = self._rho0
        is_ref = self._is_reference
        xi_fn = type(self).__dict__['_xi_fn']

        if is_ref:
            def _C_eff_pt(x):
                return C0

            def _rho_eff_pt(x):
                return rho0
        else:
            def _C_eff_pt(x):
                return C_eff(x, geo, C0)

            def _rho_eff_pt(x):
                return rho_eff(x, geo, rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp  # stored separately for set_params

        # Precompute cell mapping if cell decomposition is available
        cell_decomp = getattr(type(self), '_cell_decomp', None)
        if cell_decomp is not None:
            import numpy as np
            self._qp_to_cell = jnp.array(
                cell_decomp.build_qp_mapping(np.asarray(self.physical_quad_points))
            )

        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            xi_qp,
        ]

    def set_params(self, params):
        """Map cell-based material params to internal_vars at quadrature points.

        Parameters
        ----------
        params : (cell_C_flat, cell_rho) where
            cell_C_flat : (n_cells, n_C_params)
            cell_rho    : (n_cells,)
        """
        cell_C_flat, cell_rho = params
        cell_decomp = type(self)._cell_decomp
        _, from_flat = _get_converters(type(self)._n_C_params)

        # Convert flat params → full (2,2,2,2) tensors per cell
        cell_C_full = jax.vmap(from_flat)(cell_C_flat)  # (n_cells, 2,2,2,2)

        # Expand to quadrature points via precomputed mapping
        C_qp = cell_decomp.expand_to_quadpoints(
            cell_C_full, self._qp_to_cell, self._C0)
        rho_qp = cell_decomp.expand_to_quadpoints(
            cell_rho, self._qp_to_cell, self._rho0)

        self.internal_vars = [C_qp, rho_qp, self._xi_qp]

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
    cell_decomp: CellDecomposition | None = None,
) -> RayleighCloakProblem:
    """Assemble a ``RayleighCloakProblem`` ready for solving.

    Parameters
    ----------
    cell_decomp : optional
        If provided, enables ``set_params`` for cell-based optimisation.
    """
    C0 = C_iso(params.lam, params.mu)

    RayleighCloakProblem._omega = params.omega
    RayleighCloakProblem._geometry = geometry
    RayleighCloakProblem._is_reference = cfg.is_reference
    RayleighCloakProblem._C0 = C0
    RayleighCloakProblem._rho0 = params.rho0
    RayleighCloakProblem._xi_fn = make_xi_profile(params)
    RayleighCloakProblem._x_src = params.x_src
    RayleighCloakProblem._sigma_src = params.sigma_src
    RayleighCloakProblem._F0 = params.F0
    RayleighCloakProblem._cell_decomp = cell_decomp
    RayleighCloakProblem._n_C_params = cfg.cells.n_C_params

    problem = RayleighCloakProblem(
        mesh=mesh,
        vec=4,
        dim=2,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )
    return problem
