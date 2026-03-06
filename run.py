"""CLI entry point for running a cloaking simulation.

Usage::

    python run.py                        # uses configs/default.yaml
    python run.py configs/reference.yaml # reference (no cloak) simulation
"""

from __future__ import annotations

import sys

from rayleigh_cloak import load_config, solve
from rayleigh_cloak.io import save_npz
from rayleigh_cloak.plot import plot_results


def main(config_path: str = "configs/default.yaml") -> None:
    config = load_config(config_path)
    result = solve(config)

    save_npz(result)
    plot_results(result)

    p = result.params
    print(f"\nDone.  Domain: {p.W_total:.2f} x {p.H_total:.2f} "
          f"(physical {p.W:.2f} x {p.H:.2f})")
    print(f"  PML thickness = {p.L_pml:.3f},  xi_max = {p.xi_max},  "
          f"ramp power = {p.pml_pow}")
    print(f"  Mesh: {result.mesh.cells.shape[0]} triangles")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
    main(path)
