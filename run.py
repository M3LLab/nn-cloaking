"""CLI entry point for running a forward cloaking simulation.

Runs FEM, saves displacement plots (Re(ux), Re(uy)), and computes cloaking
loss measured two ways:
  1. Right physical boundary only
  2. All physical-domain nodes outside the cloak region

Supports both continuous C_eff (transformational) and cell-based (piecewise-
constant) material modes.

Usage::

    python run.py                           # continuous cloak (default)
    python run.py configs/continuous.yaml   # explicit continuous config
    python run.py configs/cell_based.yaml   # cell-based forward solve
    python run.py configs/reference.yaml    # reference (no cloak)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from rayleigh_cloak import load_config, solve, solve_reference
from rayleigh_cloak.io import save_npz
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import compute_cloaking_loss
from rayleigh_cloak.solver import SolutionResult, solve_cell_based, _create_geometry


# ── Plotting ──────────────────────────────────────────────────────────

def _plot_re_displacement(result: SolutionResult, output_dir: str) -> None:
    """Save Re(ux), Re(uy), and |Re(u)| plots to output_dir."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    u = result.u
    pts_x = np.asarray(result.mesh.points[:, 0])
    pts_y = np.asarray(result.mesh.points[:, 1])
    p = result.params

    os.makedirs(output_dir, exist_ok=True)

    x_off, y_off = p.x_off, p.y_off
    W, H = p.W, p.H
    phys = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8) &
            (pts_y >= y_off - 1e-8))
    px = pts_x[phys] - x_off
    py = pts_y[phys] - y_off

    # Cloak outline (physical coords)
    a, b, c_hw = p.a, p.b, p.c
    xc = W / 2.0

    fields = {
        "re_ux": ("Re(u_x)", u[:, 0]),
        "re_uy": ("Re(u_y)", u[:, 1]),
        "re_mag": ("|Re(u)|", np.sqrt(u[:, 0]**2 + u[:, 1]**2)),
    }

    for fname, (title, field) in fields.items():
        pv = field[phys]
        vlim = np.percentile(np.abs(pv), 95)
        if vlim < 1e-30:
            vlim = 1.0

        fig, ax = plt.subplots(figsize=(13, 4))
        tc = ax.tricontourf(px, py, pv, levels=100, cmap='RdBu_r',
                            vmin=-vlim, vmax=vlim)
        ax.plot(p.x_src - x_off, H, 'r*', markersize=12)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - b, H],
                ls='--', color='yellow', lw=1.2)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - a, H],
                ls='--', color='yellow', lw=1.2)
        fig.colorbar(tc, ax=ax, shrink=0.8, label=title)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')

        path = os.path.join(output_dir, f"{fname}.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    print(f"  Displacement plots saved to {output_dir}/")


# ── Main ──────────────────────────────────────────────────────────────

def main(config_path: str = "configs/continuous.yaml") -> None:
    config = load_config(config_path)
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # --- Reference (no cloak) ---
    if config.is_reference:
        print("=== Reference simulation (no cloak) ===")
        result = solve(config)
        save_npz(result)
        _plot_re_displacement(result, output_dir)
        _print_summary(result)
        return

    # --- Cloaked simulation ---
    cell_based = config.cells.enabled

    if cell_based:
        print("=== Cell-based forward solve ===")
        cloak_result = solve_cell_based(config)
        full_mesh = cloak_result.full_mesh
        kept_nodes = cloak_result.kept_nodes
    else:
        print("=== Continuous C_eff forward solve ===")
        # Generate full mesh (shared with reference), then extract submesh
        full_mesh = generate_mesh_full(config, params, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
        cloak_result = solve(config, mesh=cloak_mesh)
        cloak_result.full_mesh = full_mesh
        cloak_result.kept_nodes = kept_nodes

    # --- Reference on same full mesh ---
    print("=== Solving reference on shared mesh ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    # --- Displacement plots ---
    _plot_re_displacement(cloak_result, output_dir)

    # --- Cloaking loss ---
    loss = compute_cloaking_loss(cloak_result, ref_result, geometry)

    # --- Report ---
    mode = "cell-based" if cell_based else "continuous"
    print(f"\n{'='*60}")
    print(f"  Mode: {mode}")
    print(f"  Cloaking distortion (right boundary):  {loss.dist_right:.2f}%"
          f"  ({loss.n_right} nodes)")
    print(f"  Cloaking distortion (outside cloak):   {loss.dist_outside:.2f}%"
          f"  ({loss.n_outside} nodes)")
    print(f"{'='*60}")

    _print_summary(cloak_result)

    # Save results
    save_npz(cloak_result)


def _print_summary(result: SolutionResult) -> None:
    p = result.params
    print(f"\nDone.  Domain: {p.W_total:.2f} x {p.H_total:.2f} "
          f"(physical {p.W:.2f} x {p.H:.2f})")
    print(f"  PML thickness = {p.L_pml:.3f},  xi_max = {p.xi_max},  "
          f"ramp power = {p.pml_pow}")
    print(f"  Mesh: {result.mesh.cells.shape[0]} triangles")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/continuous.yaml"
    main(path)
