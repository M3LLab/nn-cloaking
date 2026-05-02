"""Plot histograms of lambda, mu, rho for an existing stiffness HDF5 file.

Reads the file written by ``dataset.cellular_chiral.bulk_stiffness`` and
prints summary statistics + saves a 3-panel histogram figure.

Usage
-----

    python -m dataset.cellular_chiral.dataset_stats \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/stats.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _stats(name: str, x: np.ndarray, unit: str = "") -> str:
    pcts = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return (
        f"{name:>8} ({unit:<7}) "
        f"n={x.size:7d}  "
        f"mean={x.mean():.4e}  std={x.std():.4e}  "
        f"min={x.min():.4e}  max={x.max():.4e}\n"
        f"           percentiles  1%={pcts[0]:.4e}  5%={pcts[1]:.4e}  "
        f"25%={pcts[2]:.4e}  50%={pcts[3]:.4e}  "
        f"75%={pcts[4]:.4e}  95%={pcts[5]:.4e}  99%={pcts[6]:.4e}"
    )


def _plot_hist(ax, x: np.ndarray, label: str, unit: str, log_x: bool, bins: int) -> None:
    if log_x:
        x_pos = x[x > 0]
        edges = np.logspace(np.log10(x_pos.min()), np.log10(x_pos.max()), bins + 1)
        ax.hist(x_pos, bins=edges, color="steelblue", edgecolor="white", linewidth=0.6)
        ax.set_xscale("log")
    else:
        ax.hist(x, bins=bins, color="steelblue", edgecolor="white", linewidth=0.6)

    ax.axvline(np.median(x), color="crimson", linestyle="--", linewidth=1.2,
               label=f"median = {np.median(x):.3e}")
    ax.axvline(x.mean(), color="darkorange", linestyle=":", linewidth=1.2,
               label=f"mean = {x.mean():.3e}")
    ax.set_xlabel(f"{label}  [{unit}]" if unit else label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{label}  (n={x.size})", fontsize=12)
    ax.legend(fontsize=9, loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("output/ca_bulk_squared/stiffness.h5"),
        help="Stiffness HDF5 file to read.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path for the histogram figure (default: <input dir>/stats.png).",
    )
    parser.add_argument("--bins", type=int, default=80, help="Number of histogram bins.")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Use log-scaled x-axes for lambda and mu (helpful for skewed stiffness distributions).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.input.parent / "stats.png"

    with h5py.File(args.input, "r") as f:
        lam = f["lambda_"][:]
        mu = f["mu"][:]
        rho = f["rho"][:]
        vf = f["vf"][:] if "vf" in f else None
        lf = f["live_fraction"][:] if "live_fraction" in f else None

    print(f"Loaded {lam.size} samples from {args.input}\n")

    print("Statistics")
    print("-" * 90)
    print(_stats("lambda", lam, "Pa"))
    print(_stats("mu", mu, "Pa"))
    print(_stats("rho", rho, "kg/m^3"))
    if vf is not None:
        print(_stats("vf", vf, "-"))
    if lf is not None:
        print(_stats("live_frac", lf, "-"))
    print("-" * 90)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    _plot_hist(axes[0], lam, r"$\lambda$ (Lamé)", "Pa", log_x=args.log, bins=args.bins)
    _plot_hist(axes[1], mu, r"$\mu$ (Lamé)", "Pa", log_x=args.log, bins=args.bins)
    _plot_hist(axes[2], rho, r"$\rho_{\mathrm{eff}}$", "kg/m³", log_x=False, bins=args.bins)

    fig.suptitle(f"Effective Lamé / density distributions — {args.input.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
