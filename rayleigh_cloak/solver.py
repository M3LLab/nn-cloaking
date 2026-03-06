"""High-level solve interface.

``solve(config)`` runs a full forward simulation and returns a
``SolutionResult``.  Safe to call repeatedly (e.g. inside an optimisation
loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.mesh import generate_mesh
from rayleigh_cloak.problem import build_problem


@dataclass
class SolutionResult:
    """Container for a completed simulation."""

    u: np.ndarray               # (num_nodes, 4)
    mesh: Mesh
    config: SimulationConfig
    params: DerivedParams


def _create_geometry(cfg: SimulationConfig, params: DerivedParams):
    """Instantiate the geometry object specified by ``cfg.geometry_type``."""
    if cfg.geometry_type == "triangular":
        return TriangularCloakGeometry.from_params(params)
    raise ValueError(f"Unknown geometry_type: {cfg.geometry_type!r}")


def solve(config: SimulationConfig) -> SolutionResult:
    """Run a full forward simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete specification of the problem.

    Returns
    -------
    SolutionResult
        Solution array, mesh, config, and derived parameters.
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    mesh = generate_mesh(config, params, geometry)
    problem = build_problem(mesh, config, params, geometry)

    solver_opts = {
        "petsc_solver": {
            "ksp_type": config.solver.ksp_type,
            "pc_type": config.solver.pc_type,
        }
    }

    print("Solving frequency-domain system with absorbing layers ...")
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u = sol_list[0]

    return SolutionResult(
        u=np.asarray(u),
        mesh=mesh,
        config=config,
        params=params,
    )


def solve_reference(config: SimulationConfig) -> SolutionResult:
    """Convenience: solve the reference problem (no cloak)."""
    ref_config = config.model_copy(update={"is_reference": True})
    return solve(ref_config)
