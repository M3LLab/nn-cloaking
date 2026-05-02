"""2-D mesh-convergence benchmark of the *homogenised* FEM result.

Same as ``mesh_2d_benchmark_validated.py`` but the cloak material is the
per-cell (λ, μ, ρ) coming straight out of ``optimized_params.npz`` — the
exact model the optimiser saw. No dataset matching, no pixel canvas. Only
the mesh refinement varies, so any mesh-driven non-convergence is a property
of the FEM discretisation itself, not of any post-hoc snapping.

Usage
-----

    PYTHONPATH=/home/m3l/workspace/nn-cloaking \\
    python scripts/mesh_2d_benchmark_homogenised.py \\
        configs/multifreq_small.yaml \\
        output/multifreq_small/optimized_params.npz \\
        --f-star 2.0 \\
        --cloak 5,10,15,25,35,50 \\
        --outside 1.0,0.5,0.25
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import jax.numpy as jnp                              # noqa: E402
import jax_fem.solver                                # noqa: E402

from rayleigh_cloak import load_config               # noqa: E402
from rayleigh_cloak.cells import CellDecomposition   # noqa: E402
from rayleigh_cloak.config import DerivedParams      # noqa: E402
from rayleigh_cloak.loss import (                                # noqa: E402
    make_fixed_surface_eval_points,
    transmitted_displacement_ratio,
    transmitted_displacement_ratio_fixed,
)
from rayleigh_cloak.materials import C_iso           # noqa: E402
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full  # noqa: E402
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices  # noqa: E402
from rayleigh_cloak.problem import build_problem     # noqa: E402
from rayleigh_cloak.solver import _create_geometry, solve_reference  # noqa: E402

import logging                                        # noqa: E402
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _make_config(
    base_config,
    f_star: float,
    rf_cloak: float,
    rf_outside: float,
    n_eval_points: int | None = None,
    eval_noise_sigma: float | None = None,
    eval_noise_seed: int | None = None,
    embed_macro_grid: bool | None = None,
):
    loss_updates = {}
    if n_eval_points is not None:
        loss_updates["n_eval_points"] = int(n_eval_points)
    if eval_noise_sigma is not None:
        loss_updates["eval_noise_sigma"] = float(eval_noise_sigma)
    if eval_noise_seed is not None:
        loss_updates["eval_noise_seed"] = int(eval_noise_seed)
    mesh_updates = {
        "refinement_factor_cloak":   float(rf_cloak),
        "refinement_factor_outside": float(rf_outside),
    }
    if embed_macro_grid is not None:
        mesh_updates["embed_macro_grid"] = bool(embed_macro_grid)
    update = {
        "domain": base_config.domain.model_copy(update={"f_star": float(f_star)}),
        "mesh":   base_config.mesh.model_copy(update=mesh_updates),
    }
    if loss_updates:
        update["loss"] = base_config.loss.model_copy(update=loss_updates)
    return base_config.model_copy(update=update)


def _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes):
    cs_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, dp.x_off, dp.x_off + dp.W,
    )
    return cs_idx, kept_nodes[cs_idx]


def _format_grid(metric_name: str, cloaks, outsides, grid: dict[tuple, str]) -> str:
    col_w = max(8, max(len(g) for g in grid.values()) + 1)
    head = f"{metric_name:>16}  " + "  ".join(f"out={o:>5}".rjust(col_w) for o in outsides)
    sep = "-" * len(head)
    lines = [head, sep]
    for c in cloaks:
        row = f"clk={c:<5}".rjust(16) + "  " + "  ".join(
            f"{grid[(c, o)]:>{col_w}}" for o in outsides
        )
        lines.append(row)
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config")
    p.add_argument("params")
    p.add_argument("--f-star", type=float, default=2.0)
    p.add_argument("--cloak", default="5,10,15,25,35,50")
    p.add_argument("--outside", default="1.0,0.5,0.25")
    p.add_argument("-o", "--output-dir", default=None)
    p.add_argument("--n-eval-points", type=int, default=None,
                   help="Override loss.n_eval_points. 0 keeps the legacy "
                        "node-based metric; >0 evaluates |u| at this many "
                        "fixed x-positions (mesh-independent).")
    p.add_argument("--eval-noise-sigma", type=float, default=None,
                   help="Override loss.eval_noise_sigma (Gaussian jitter on "
                        "the fixed x-positions, in physical units).")
    p.add_argument("--eval-noise-seed", type=int, default=None,
                   help="Override loss.eval_noise_seed.")
    p.add_argument("--embed-macro-grid", action="store_true",
                   help="Embed the (n_x-1)+(n_y-1) interior macro-grid lines as "
                        "1-D constraints in the gmsh surface so no FEM element "
                        "straddles a macro-cell boundary.")
    args = p.parse_args()

    cloaks   = [float(x.strip()) for x in args.cloak.split(",")    if x.strip()]
    outsides = [float(x.strip()) for x in args.outside.split(",") if x.strip()]
    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"mesh_2d_benchmark_homogenised_f{args.f_star:.2f}.csv"

    # Load optimised params (kept identical across all mesh resolutions).
    npz = np.load(args.params)
    cell_C_flat = jnp.asarray(npz["cell_C_flat"])
    cell_rho = jnp.asarray(npz["cell_rho"])
    opt_params = (cell_C_flat, cell_rho)
    print(f"loaded params: cell_C_flat {tuple(cell_C_flat.shape)}, "
          f"cell_rho {tuple(cell_rho.shape)}")

    solver_opts = {
        "petsc_solver": {
            "ksp_type": base_config.solver.ksp_type,
            "pc_type": base_config.solver.pc_type,
        }
    }

    # CellDecomposition is geometry-dependent; geometry is mesh-independent
    # (only depends on the *physical* cloak description), so we can build it
    # once outside the sweep. The dp/geometry inside the loop are rebuilt
    # because f_star may change them (in fact it doesn't here; both depend
    # only on dimensionless geometry factors).
    print(f"sweeping {len(cloaks)} × {len(outsides)} = {len(cloaks)*len(outsides)} cells")

    # Write the CSV header up front and append after each sweep point so an
    # OOM mid-sweep doesn't lose all previous results.
    with open(csv_path, "w") as fh:
        fh.write("rf_cloak,rf_outside,nodes,cells,ratio,wall_s,peak_rss_gb,status\n")

    rows: list[dict] = []
    for c in cloaks:
        for o in outsides:
            cfg = _make_config(
                base_config, args.f_star, c, o,
                n_eval_points=args.n_eval_points,
                eval_noise_sigma=args.eval_noise_sigma,
                eval_noise_seed=args.eval_noise_seed,
                embed_macro_grid=(True if args.embed_macro_grid else None),
            )
            dp = DerivedParams.from_config(cfg)
            geometry = _create_geometry(cfg, dp)
            cell_decomp = CellDecomposition(
                geometry, base_config.cells.n_x, base_config.cells.n_y,
            )

            # Build the fixed evaluation x-positions ONCE per (c, o) pair
            # (they only depend on geometry/dp, which are constant across the
            # sweep at fixed f_star — but rebuilding is cheap and clearer).
            eval_xs = None
            if cfg.loss.n_eval_points > 0:
                eval_xs = make_fixed_surface_eval_points(
                    geometry, dp,
                    cfg.loss.n_eval_points,
                    cfg.loss.eval_noise_sigma,
                    cfg.loss.eval_noise_seed,
                )

            t0 = time.time()
            try:
                full_mesh = generate_mesh_full(cfg, dp, geometry)
                cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
                n_nodes = len(cloak_mesh.points)
                n_cells = int(cloak_mesh.cells.shape[0])
                print(
                    f"  rf_cloak={c:>5}  rf_out={o:>5}  "
                    f"nodes={n_nodes:>7}  cells={n_cells:>8}  ...",
                    end="", flush=True,
                )

                ref_result = solve_reference(cfg, mesh=full_mesh)
                problem = build_problem(cloak_mesh, cfg, dp, geometry, cell_decomp)
                problem.set_params(opt_params)
                sol_list = jax_fem.solver.solver(problem, solver_options=solver_opts)
                u_opt = np.asarray(sol_list[0])
                if eval_xs is not None:
                    ratio = transmitted_displacement_ratio_fixed(
                        u_opt, ref_result.u, cloak_mesh, full_mesh,
                        eval_xs, dp.y_top,
                    )
                else:
                    cs_idx, rs_idx = _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes)
                    ratio = float(transmitted_displacement_ratio(u_opt, ref_result.u, cs_idx, rs_idx))
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(f"ratio={ratio:.4f}  wall={wall:.1f}s  rss={rss:.2f} GB")
                row = {
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": n_nodes, "cells": n_cells,
                    "ratio": ratio, "wall_s": wall, "peak_rss_gb": rss,
                    "status": "ok",
                }
            except Exception as exc:                                # noqa: BLE001
                wall = time.time() - t0
                rss = _peak_rss_gb()
                print(f"\n  FAILED: {type(exc).__name__}: {exc}")
                row = {
                    "rf_cloak": c, "rf_outside": o,
                    "nodes": -1, "cells": -1,
                    "ratio": float("nan"), "wall_s": wall, "peak_rss_gb": rss,
                    "status": f"fail:{type(exc).__name__}",
                }
            rows.append(row)
            with open(csv_path, "a") as fh:
                fh.write(
                    f"{row['rf_cloak']},{row['rf_outside']},{row['nodes']},{row['cells']},"
                    f"{row['ratio']:.6f},{row['wall_s']:.1f},{row['peak_rss_gb']:.2f},"
                    f"{row['status']}\n"
                )

    print(f"\nCSV → {csv_path}")

    by_pair = {(r["rf_cloak"], r["rf_outside"]): r for r in rows}
    print(f"\n=== {Path(args.params).parent.name}  f*={args.f_star:.2f} (homogenised) ===")
    print(_format_grid(
        "u_ratio", cloaks, outsides,
        {k: f"{v['ratio']:.4f}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "cells", cloaks, outsides,
        {k: f"{v['cells']:>7}" if v["status"] == "ok" else "FAIL" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "wall_s", cloaks, outsides,
        {k: f"{v['wall_s']:.1f}" for k, v in by_pair.items()},
    ))
    print()
    print(_format_grid(
        "rss_gb", cloaks, outsides,
        {k: f"{v['peak_rss_gb']:.2f}" for k, v in by_pair.items()},
    ))


if __name__ == "__main__":
    main()
