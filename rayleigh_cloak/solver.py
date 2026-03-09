"""High-level solve interface.

``solve(config)`` runs a full forward simulation and returns a
``SolutionResult``.  Safe to call repeatedly (e.g. inside an optimisation
loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper, solver as jax_fem_solver

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh, generate_mesh_full
from rayleigh_cloak.optimize import (
    OptimizationResult,
    get_right_boundary_indices,
    run_optimization,
)
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


def solve(
    config: SimulationConfig,
    mesh: Mesh | None = None,
) -> SolutionResult:
    """Run a full forward simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete specification of the problem.
    mesh : Mesh, optional
        Pre-built mesh.  If *None*, a mesh is generated from the config.

    Returns
    -------
    SolutionResult
        Solution array, mesh, config, and derived parameters.
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    if mesh is None:
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


def solve_reference(
    config: SimulationConfig,
    mesh: Mesh | None = None,
) -> SolutionResult:
    """Convenience: solve the reference problem (no cloak)."""
    ref_config = config.model_copy(update={"is_reference": True})
    return solve(ref_config, mesh=mesh)


def solve_optimization(config: SimulationConfig) -> OptimizationResult:
    """Run cell-based material optimisation.

    Steps:
    1. Generate a single full-domain mesh (no defect cutout).
    2. Solve the reference problem on the full mesh.
    3. Extract submesh (defect elements removed) for the cloak solve.
    4. Build cell decomposition and FEM problem on the submesh.
    5. Optimise cell materials — boundary nodes match exactly (no interpolation).
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = generate_mesh_full(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    print("=== Step 3: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements "
          f"({len(full_mesh.cells) - len(cloak_mesh.cells)} defect elements removed, "
          f"{len(full_mesh.points) - len(kept_nodes)} orphan nodes removed)")

    print("=== Step 4: Setting up cell decomposition ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
    )

    print(f"  {cell_decomp.n_cells} total cells, "
          f"{cell_decomp.n_cloak_cells} in cloak")

    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)

    # Right physical boundary on the submesh
    x_right = params.x_off + params.W
    boundary_indices = get_right_boundary_indices(
        np.asarray(cloak_mesh.points), x_right)
    print(f"  {len(boundary_indices)} boundary nodes for loss")

    # Map boundary indices back to full-mesh numbering for reference solution
    u_ref_boundary = ref_result.u[kept_nodes[boundary_indices]]
    print(f"  Extracted u_ref at {len(boundary_indices)} boundary nodes")

    neighbor_pairs = cell_decomp.get_neighbor_pairs()
    print(f"  {len(neighbor_pairs)} neighbor pairs for regularisation")

    print("=== Step 5: Optimising ===")
    solver_opts = {
        "petsc_solver": {
            "ksp_type": config.solver.ksp_type,
            "pc_type": config.solver.pc_type,
        }
    }
    adjoint_opts = {
        "petsc_solver": {"ksp_type": "gmres", "pc_type": "ilu"}
    }
    fwd_pred = ad_wrapper(problem, solver_opts, adjoint_opts)
    params_init = cell_mat.get_initial_params()

    # Per-step displacement plot callback
    pts_x = np.asarray(cloak_mesh.points[:, 0])
    pts_y = np.asarray(cloak_mesh.points[:, 1])
    step_dir = f"{config.output_dir}/opt_steps"

    def _plot_step(step: int, u: np.ndarray) -> None:
        from rayleigh_cloak.plot import plot_displacement_field
        plot_displacement_field(
            u, pts_x, pts_y, params,
            save_path=f"{step_dir}/step_{step:04d}.png",
            title=f"|Re(u)| — step {step}",
        )

    opt_cfg = config.optimization
    result = run_optimization(
        fwd_pred=fwd_pred,
        params_init=params_init,
        u_ref_boundary=u_ref_boundary,
        boundary_indices=boundary_indices,
        neighbor_pairs=neighbor_pairs,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lambda_l2=opt_cfg.lambda_l2,
        lambda_neighbor=opt_cfg.lambda_neighbor,
        plot_callback=_plot_step if opt_cfg.plot_every > 0 else None,
        plot_every=opt_cfg.plot_every,
    )
    return result
