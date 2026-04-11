"""Frequency sweep: evaluate cloaking performance vs f_star.

Implements the metric from Chatzopoulos et al. (2023), Fig 2(k):
  <|u_cloak|> / <|u_ref|>
i.e. the ratio of the average total displacement magnitude on the
free surface beyond the cloaked region.  A perfect cloak → 1.0.

The domain geometry, mesh, and PML are independent of f_star — only ω
changes — so they are built once and reused across all frequencies.

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
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import SolutionResult, _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── metric helpers ────────────────────────────────────────────────────


def _displacement_magnitude(u: np.ndarray) -> np.ndarray:
    """Total displacement magnitude per node: sqrt(|ux|^2 + |uy|^2).

    u has shape (n_nodes, 4) with DOFs [Re(ux), Re(uy), Im(ux), Im(uy)].
    """
    return np.sqrt(u[:, 0]**2 + u[:, 1]**2 + u[:, 2]**2 + u[:, 3]**2)


def _get_top_surface_beyond_cloak(
    mesh_points: np.ndarray,
    y_top: float,
    x_c: float,
    c: float,
    x_right: float,
    tol: float = 1e-6,
) -> np.ndarray:
    """Node indices on the top surface downstream (right) of the cloak.

    Selects nodes with y ≈ y_top and x > x_c + c (beyond the cloak
    half-width) and x ≤ x_right (within the physical domain, excluding PML).
    """
    pts = np.asarray(mesh_points)
    on_top = np.abs(pts[:, 1] - y_top) < tol
    beyond_cloak = pts[:, 0] > (x_c + c + tol)
    within_phys = pts[:, 0] <= (x_right + tol)
    return np.where(on_top & beyond_cloak & within_phys)[0]


def transmitted_displacement_ratio(
    u_cloak: np.ndarray,
    u_ref: np.ndarray,
    cloak_idx: np.ndarray,
    ref_idx: np.ndarray,
) -> float:
    """⟨|u_cloak|⟩ / ⟨|u_ref|⟩  on the surface beyond the cloak.

    This is the metric from Chatzopoulos et al. (2023), Fig 2(k).
    """
    mag_cloak = _displacement_magnitude(u_cloak[cloak_idx])
    mag_ref = _displacement_magnitude(u_ref[ref_idx])
    avg_cloak = float(np.mean(mag_cloak))
    avg_ref = float(np.mean(mag_ref)) + 1e-30
    return avg_cloak / avg_ref


# ── plotting ──────────────────────────────────────────────────────────


def plot_results(csv_path: Path, out_dir: Path) -> None:
    """Plot from an existing CSV."""
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    f_vals = data["f_star"]
    ratio_vals = data["u_ratio"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_vals, ratio_vals, "o-", color="C0", lw=1.5, markersize=5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u_{\rm cloak}| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Cloaking performance vs frequency")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_vals.max() + 0.1)
    ax.set_ylim(0, max(ratio_vals.max() * 1.1, 1.15))

    plot_path = out_dir / "frequency_sweep.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
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

    # ── Build mesh and geometry ONCE ──
    # f_star only affects omega; domain size, cloak geometry, PML, mesh
    # all depend on lambda_star (which is fixed).
    dp_base = DerivedParams.from_config(base_config)
    geometry = _create_geometry(base_config, dp_base)

    print("=== Generating mesh (reused across all frequencies) ===")
    full_mesh = generate_mesh_full(base_config, dp_base, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    # Set up cell decomposition (geometry-dependent, not frequency-dependent)
    cell_decomp = CellDecomposition(geometry, base_config.cells.n_x, base_config.cells.n_y)
    C0 = C_iso(dp_base.lam, dp_base.mu)
    CellMaterial(
        geometry, C0, dp_base.rho0, cell_decomp,
        n_C_params=base_config.cells.n_C_params,
    )

    # ── Identify evaluation nodes ──
    # Top surface beyond cloak on the cloak mesh
    x_right_phys = dp_base.x_off + dp_base.W
    cloak_surface_idx = _get_top_surface_beyond_cloak(
        cloak_mesh.points, dp_base.y_top, dp_base.x_c, dp_base.c,
        x_right_phys,
    )
    # Corresponding nodes on the full mesh (for reference solution)
    ref_surface_idx = kept_nodes[cloak_surface_idx]
    print(f"Surface evaluation nodes beyond cloak: {len(cloak_surface_idx)}")

    # ── Frequency sweep ──
    f_stars = np.arange(0.1, 4.05, 0.1)
    results = []

    for f_star in f_stars:
        print(f"\n{'='*60}")
        print(f"f_star = {f_star:.1f}")
        print(f"{'='*60}")

        # Only omega changes — rebuild DerivedParams and Problem, not mesh
        config = base_config.model_copy(
            update={"domain": base_config.domain.model_copy(update={"f_star": float(f_star)})}
        )
        dp = DerivedParams.from_config(config)

        # Reference solve (same mesh, new omega)
        ref_result = solve_reference(config, mesh=full_mesh)

        # Cloaked solve with optimized params (same submesh, new omega)
        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params(opt_params)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_cloak = np.asarray(sol_list[0])

        # Paper metric: <|u_cloak|> / <|u_ref|>
        u_ratio = transmitted_displacement_ratio(
            u_cloak, ref_result.u, cloak_surface_idx, ref_surface_idx,
        )
        print(f"  u_ratio = {u_ratio:.4f}  ({len(cloak_surface_idx)} surface nodes)")

        results.append({
            "f_star": float(f_star),
            "u_ratio": u_ratio,
        })

    # Save CSV
    csv_path = out_dir / "frequency_sweep.csv"
    with open(csv_path, "w") as f:
        f.write("f_star,u_ratio\n")
        for r in results:
            f.write(f"{r['f_star']:.1f},{r['u_ratio']:.6f}\n")
    print(f"\nCSV saved to {csv_path}")

    plot_results(csv_path, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Frequency sweep for cloaking performance")
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
