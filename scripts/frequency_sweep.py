"""Frequency sweep: evaluate cloaking distortion vs f_star.

Loads optimized cell parameters and sweeps f_star from 0.1 to 4.0,
computing the right-boundary cloaking distortion at each frequency.

By default, plots from an existing CSV. Use -f to force re-running solves.

Usage::

    # Plot from existing results:
    python scripts/frequency_sweep.py \
        configs/triangular_optimize_neural_flat2.yaml \
        output/triangular_optimize_neural_flat2/optimized_params.npz

    # Force re-run solves (overwrites CSV):
    python scripts/frequency_sweep.py -f \
        configs/triangular_optimize_neural_flat2.yaml \
        output/triangular_optimize_neural_flat2/optimized_params.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import compute_cloaking_loss
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import SolutionResult, _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def plot_results(csv_path: Path, out_dir: Path) -> None:
    """Plot from an existing CSV."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    f_vals = data["f_star"]
    dist_vals = data["dist_right"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_vals, dist_vals, "o-", color="C0", lw=1.5, markersize=5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel("Right-boundary distortion (%)")
    ax.set_title("Cloaking performance vs frequency")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_vals.max() + 0.1)
    ax.set_ylim(bottom=0)

    plot_path = out_dir / "frequency_sweep.png"
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def run_sweep(config_path: str, params_path: str) -> None:
    base_config = load_config(config_path)
    out_dir = Path(base_config.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Load optimized parameters
    data = np.load(params_path)
    cell_C_flat = jnp.array(data["cell_C_flat"])
    cell_rho = jnp.array(data["cell_rho"])
    opt_params = (cell_C_flat, cell_rho)
    print(f"Loaded params: cell_C_flat {cell_C_flat.shape}, cell_rho {cell_rho.shape}")

    solver_opts = {
        "petsc_solver": {
            "ksp_type": base_config.solver.ksp_type,
            "pc_type": base_config.solver.pc_type,
        }
    }

    f_stars = np.arange(0.1, 4.05, 0.1)
    results = []

    for f_star in f_stars:
        print(f"\n{'='*60}")
        print(f"f_star = {f_star:.1f}")
        print(f"{'='*60}")

        # Changing f_star changes omega AND lambda_star-derived dimensions
        # (domain size, PML, geometry, source width, mesh), so rebuild everything.
        config = base_config.model_copy(
            update={"domain": base_config.domain.model_copy(update={"f_star": float(f_star)})}
        )
        dp = DerivedParams.from_config(config)
        geometry = _create_geometry(config, dp)

        # Generate mesh and reference solution
        full_mesh = generate_mesh_full(config, dp, geometry)
        ref_result = solve_reference(config, mesh=full_mesh)

        # Extract submesh (remove defect)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

        # Set up cell decomposition
        cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
        C0 = C_iso(dp.lam, dp.mu)
        CellMaterial(
            geometry, C0, dp.rho0, cell_decomp,
            n_C_params=config.cells.n_C_params,
        )

        # Forward solve with optimized params
        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params(opt_params)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_cloak = np.asarray(sol_list[0])

        cloak_result = SolutionResult(
            u=u_cloak,
            mesh=cloak_mesh,
            config=config,
            params=dp,
            full_mesh=full_mesh,
            kept_nodes=kept_nodes,
        )

        loss = compute_cloaking_loss(cloak_result, ref_result, geometry)
        print(f"  dist_right = {loss.dist_right:.2f}%  ({loss.n_right} nodes)")
        results.append({
            "f_star": float(f_star),
            "dist_right": loss.dist_right,
            "dist_boundary": loss.dist_boundary,
            "dist_outside": loss.dist_outside,
            "n_right": loss.n_right,
        })

    # Save CSV
    csv_path = out_dir / "frequency_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("f_star,dist_right,dist_boundary,dist_outside,n_right\n")
        for r in results:
            f.write(f"{r['f_star']:.1f},{r['dist_right']:.4f},"
                    f"{r['dist_boundary']:.4f},{r['dist_outside']:.4f},"
                    f"{r['n_right']}\n")
    print(f"\nCSV saved to {csv_path}")

    plot_results(csv_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Frequency sweep for cloaking distortion")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("params", help="Path to optimized_params.npz")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-run solves (overwrite existing CSV)")
    args = parser.parse_args()

    out_dir = Path(load_config(args.config).output_dir)
    csv_path = out_dir / "frequency_sweep.csv"

    if not args.force and csv_path.exists():
        print(f"Found existing {csv_path}, plotting from CSV. Use -f to re-run solves.")
        plot_results(csv_path, out_dir)
    else:
        run_sweep(args.config, args.params)


if __name__ == "__main__":
    main()
