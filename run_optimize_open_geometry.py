"""CLI entry point for joint shape + material optimisation.

Initialises a per-cell shape mask from the triangular cloak, then lets both
the mask (pixel occupancy) and the underlying cell material evolve together.
The cloak shape is therefore free to differ from the triangle, while the
physics pipeline is reused unchanged.

Usage::

    python run_optimize_open_geometry.py                                  # configs/open_geometry.yaml
    python run_optimize_open_geometry.py configs/open_geometry.yaml       # explicit
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak import load_config
from rayleigh_cloak.open_geometry import (
    ShapeOptConfig,
    solve_optimization_open_geometry,
)
from rayleigh_cloak.shape_mask import largest_connected_component

logging.getLogger("jax_fem").setLevel(logging.WARNING)


def _plot_loss(result, save_path: Path) -> None:
    steps = np.arange(len(result.loss_history))
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ax1, ax2, ax3 = axes

    ax1.semilogy(steps, result.loss_history, "k-", lw=1.5, label="total")
    ax1.semilogy(steps, result.cloak_history, "C0-", lw=1, label="cloak")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(steps, result.l2_history, "C1-", lw=1, label="L2 drift")
    ax2.semilogy(steps, result.neighbor_history, "C2-", lw=1, label="material TV")
    ax2.semilogy(steps, result.mask_smooth_history, "C3-", lw=1, label="mask TV")
    if any(v > 0 for v in result.bin_history):
        ax2.semilogy(steps, result.bin_history, "C4-", lw=1, label="binarisation")
    ax2.set_ylabel("Regularisation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    if result.beta_history and max(result.beta_history) > min(result.beta_history):
        ax3.plot(steps, result.beta_history, "C5-", lw=1.5, label="β")
        ax3.set_ylabel("β (sigmoid sharpness)")
        ax3.set_xlabel("Step")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        # β constant — hide the panel
        ax3.set_visible(False)
        ax2.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _plot_mask(mask_grid: np.ndarray, save_path: Path, step: int | None = None,
               title_suffix: str = "", cmap: str = "magma",
               vmin: float = 0.0, vmax: float = 1.0,
               label: str = "mask = sigmoid(β·logit)") -> None:
    """Plot a ``(n_x, n_y)`` mask array as a heatmap."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.pcolormesh(mask_grid.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_aspect("equal")
    title = "Shape mask" + title_suffix
    if step is not None:
        title += f" — step {step}"
    ax.set_title(title)
    ax.set_xlabel("cell ix")
    ax.set_ylabel("cell iy")
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


def main(config_path: str = "configs/open_geometry.yaml") -> None:
    config = load_config(config_path)
    shape_cfg = ShapeOptConfig.from_yaml(config_path)

    if not config.cells.enabled:
        print("Error: cells.enabled must be true for open-geometry optimisation.")
        sys.exit(1)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out / "config.yaml")

    mask_dir = out / "shape_steps"
    if shape_cfg.plot_mask_every > 0:
        mask_dir.mkdir(exist_ok=True)

    loss_csv = open(out / "loss_history.csv", "w")
    loss_csv.write(
        "step,beta,total,cloak,l2,material_tv,mask_tv,bin,mask_solid_frac\n"
    )
    loss_csv.flush()

    best_loss = [float("inf")]

    def _step_cb(step, total, cloak, l2, nb, tv, bin_, beta_t, state):
        import jax
        from rayleigh_cloak.shape_mask import smooth_logits
        # Effective mask mirrors what the FEM actually sees
        s_eff = smooth_logits(
            state["logits"], config.cells.n_x, config.cells.n_y,
            shape_cfg.smooth_sigma,
        )
        m = np.asarray(jax.nn.sigmoid(beta_t * s_eff))
        solid_frac = float((m > 0.5).mean())
        loss_csv.write(
            f"{step},{beta_t:.4f},{total:.8e},{cloak:.8e},"
            f"{l2:.8e},{nb:.8e},{tv:.8e},{bin_:.8e},{solid_frac:.6f}\n"
        )
        loss_csv.flush()
        if total < best_loss[0]:
            best_loss[0] = total
            np.savez(
                out / "optimized_params.npz",
                cell_C_flat=np.asarray(state["C"]),
                cell_rho=np.asarray(state["rho"]),
                shape_logits=np.asarray(state["logits"]),
                shape_mask=m,
                beta=float(beta_t),
                n_x=config.cells.n_x,
                n_y=config.cells.n_y,
            )
            print(f"    ✓ new best loss {total:.4e} at step {step}")

    def _mask_cb(step, mask_grid):
        path = mask_dir / f"mask_{step:04d}.png"
        _plot_mask(mask_grid, path, step=step)

    try:
        result = solve_optimization_open_geometry(
            config=config,
            shape_cfg=shape_cfg,
            step_callback=_step_cb,
            mask_callback=_mask_cb if shape_cfg.plot_mask_every > 0 else None,
        )
    finally:
        loss_csv.close()

    print(f"\nOptimisation done. {len(result.loss_history)} iterations.")
    print(f"  Loss: {result.loss_history[0]:.4e} → {result.loss_history[-1]:.4e}")
    print(f"  Best loss: {best_loss[0]:.4e}")

    final_path = out / "optimized_params_final.npz"
    np.savez(
        final_path,
        cell_C_flat=np.asarray(result.cell_C_flat),
        cell_rho=np.asarray(result.cell_rho),
        shape_logits=np.asarray(result.shape_logits),
        shape_mask=np.asarray(result.shape_mask),
        beta=result.beta_end,
        n_x=result.n_x,
        n_y=result.n_y,
    )
    print(f"  Final params: {final_path}")

    final_mask_grid = np.asarray(result.shape_mask).reshape(result.n_x, result.n_y)
    _plot_mask(final_mask_grid, out / "shape_mask_final.png", title_suffix=" (final)")
    print(f"  Final mask plot: {out / 'shape_mask_final.png'}")

    _plot_loss(result, out / "loss_curves.pdf")
    print(f"  Loss plot: {out / 'loss_curves.pdf'}")

    # Post-hoc: largest connected component — a manufacturable single-body mask
    if shape_cfg.project_final:
        projected = largest_connected_component(
            final_mask_grid, connectivity=shape_cfg.project_connectivity,
        )
        n_solid = int(projected.sum())
        n_discarded = int((final_mask_grid > 0.5).sum()) - n_solid
        print(
            f"  Projected mask: largest component keeps {n_solid} cells, "
            f"discards {n_discarded} (conn={shape_cfg.project_connectivity})"
        )
        np.savez(
            out / "optimized_params_projected.npz",
            cell_C_flat=np.asarray(result.cell_C_flat),
            cell_rho=np.asarray(result.cell_rho),
            shape_logits=np.asarray(result.shape_logits),
            shape_mask=projected.ravel().astype(np.float32),
            shape_mask_soft=np.asarray(result.shape_mask),
            beta=result.beta_end,
            n_x=result.n_x,
            n_y=result.n_y,
            project_connectivity=shape_cfg.project_connectivity,
        )
        _plot_mask(
            projected.astype(float), out / "shape_mask_projected.png",
            title_suffix=" (projected, largest CC)",
            cmap="gray_r", label="mask ∈ {0, 1}",
        )
        print(f"  Projected artefacts: "
              f"{out / 'optimized_params_projected.npz'} + "
              f"{out / 'shape_mask_projected.png'}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/open_geometry.yaml"
    main(path)
