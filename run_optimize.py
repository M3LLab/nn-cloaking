"""CLI entry point for cell-based material optimisation.

Usage::

    python run_optimize.py                        # uses configs/cell_based.yaml
    python run_optimize.py configs/optimize.yaml  # custom config
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak import load_config
from rayleigh_cloak.solver import (
    solve_optimization,
    solve_optimization_neural,
    solve_optimization_neural_topo,
)
import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def plot_loss(result, save_path: str) -> None:
    """Plot total and component loss curves."""
    steps = np.arange(len(result.loss_history))
    has_neighbor = hasattr(result, "neighbor_history") and result.neighbor_history

    if has_neighbor:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

    ax1.semilogy(steps, result.loss_history, "k-", lw=1.5, label="total")
    ax1.semilogy(steps, result.cloak_history, "C0-", lw=1, label="cloak")
    ax1.semilogy(steps, result.l2_history, "C1-", lw=1, label="L2 reg")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if has_neighbor:
        ax2.semilogy(steps, result.l2_history, "C1-", lw=1, label="L2 reg")
        ax2.semilogy(steps, result.neighbor_history, "C2-", lw=1, label="neighbor")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Regularisation")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax1.set_xlabel("Step")

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str = "configs/cell_based.yaml") -> None:
    config = load_config(config_path)

    if not config.cells.enabled:
        print("Error: cells.enabled must be true for optimisation. "
              "Use run.py for forward solves.")
        sys.exit(1)

    out = Path(config.output_dir)
    out.mkdir(exist_ok=True)

    # Open CSV for incremental writing
    loss_csv = out / "loss_history.csv"
    csv_file = open(loss_csv, "w")
    method = config.optimization.method
    if method == "neural_topo":
        csv_file.write("step,total,cloak,l2_reg,neighbor,vol_frac,C_rel_err,rho_rel_err\n")
    else:
        csv_file.write("step,total,cloak,l2_reg,neighbor\n")
    csv_file.flush()

    best_loss = float("inf")

    def _log_step(step, total, cloak, l2, neighbor, params, mat_metrics=None):
        nonlocal best_loss
        if mat_metrics is not None:
            csv_file.write(
                f"{step},{total:.8e},{cloak:.8e},{l2:.8e},{neighbor:.8e},"
                f"{mat_metrics['vol_frac']:.6f},{mat_metrics['C_rel_err']:.6f},"
                f"{mat_metrics['rho_rel_err']:.6f}\n"
            )
        else:
            csv_file.write(f"{step},{total:.8e},{cloak:.8e},{l2:.8e},{neighbor:.8e}\n")
        csv_file.flush()
        if total < best_loss:
            best_loss = total
            cell_C_flat, cell_rho = params
            np.savez(out / "optimized_params.npz",
                     cell_C_flat=np.asarray(cell_C_flat),
                     cell_rho=np.asarray(cell_rho))
            print(f"    ✓ New best loss {total:.4e} at step {step}, saved params")

    try:
        if method == "neural_topo":
            print(f"Using topology neural reparameterization (method={method})")
            result = solve_optimization_neural_topo(config, step_callback=_log_step)
        elif method == "neural":
            print(f"Using neural reparameterization (method={method})")
            result = solve_optimization_neural(config, step_callback=_log_step)
        else:
            print(f"Using raw parameter optimization (method={method})")
            result = solve_optimization(config, step_callback=_log_step)
    finally:
        csv_file.close()

    print(f"\nOptimisation done. {len(result.loss_history)} iterations.")
    print(f"  Loss: {result.loss_history[0]:.4e} → {result.loss_history[-1]:.4e}")
    print(f"  Best loss: {best_loss:.4e} (saved to {out / 'optimized_params.npz'})")
    print(f"  Loss log: {loss_csv}")

    # Plot loss curves
    plot_loss(result, save_path=str(out / "loss_curves.pdf"))
    print(f"  Loss plot: {out / 'loss_curves.pdf'}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/cell_based.yaml"
    main(path)
