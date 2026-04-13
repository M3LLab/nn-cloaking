#!/usr/bin/env python
"""Generate a surrogate-model dataset: (cell_params, f_star) → cloaking loss.

Produces an HDF5 file with random samples and optimization-trajectory
snapshots across a range of frequencies.  The geometry (mesh, cell grid,
cloak_mask) is fixed from the base config; only material parameters and
frequency vary.

Default: ~10k samples (60% near-optimal from optimization trajectories).
79 frequencies (0.1–4.0, step 0.05) × (76 opt snapshots + 25 random +
25 smooth + 1 init) = ~10,033 samples.

Usage::

    # Default settings (~10k samples, 79 frequencies, ~60% optimization)
    python scripts/generate_surrogate_dataset.py configs/surrogate_dataset.yaml

    # Custom frequency range and sample counts
    python scripts/generate_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        --f-min 0.5 --f-max 3.0 --f-step 0.1 \\
        --n-random 20 --opt-iters 50

    # Quick test run (2 frequencies, few samples)
    python scripts/generate_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        --f-min 1.0 --f-max 2.0 --f-step 1.0 \\
        --n-random 3 --n-smooth 3 --opt-iters 10

    # Only random + smooth samples (no optimization trajectory)
    python scripts/generate_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        --opt-iters 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

# Suppress noisy jax-fem logging
import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)

from rayleigh_cloak import load_config
from dataset_gen.generate import DatasetGenConfig, run_dataset_generation


def main():
    parser = argparse.ArgumentParser(
        description="Generate surrogate dataset for cloaking loss prediction",
    )
    parser.add_argument(
        "config",
        help="Base YAML config (defines geometry, mesh, cells, loss type)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output HDF5 path (default: {output_dir}/surrogate_dataset.h5)",
    )

    # Frequency grid (default: 79 freqs, narrow 0.05 steps)
    parser.add_argument("--f-min", type=float, default=0.1,
                        help="Minimum f_star (default: 0.1)")
    parser.add_argument("--f-max", type=float, default=4.0,
                        help="Maximum f_star (default: 4.0)")
    parser.add_argument("--f-step", type=float, default=0.05,
                        help="f_star step size (default: 0.05)")
    parser.add_argument("--f-stars", type=float, nargs="+", default=None,
                        help="Explicit list of f_star values (overrides min/max/step)")

    # Random perturbation samples
    parser.add_argument("--n-random", type=int, default=25,
                        help="Random perturbation samples per frequency (default: 25)")
    parser.add_argument("--noise-scales", type=float, nargs="+",
                        default=[0.01, 0.05, 0.1, 0.2, 0.5],
                        help="Relative noise magnitudes to cycle through")

    # Smooth random field samples
    parser.add_argument("--n-smooth", type=int, default=25,
                        help="Smooth random field samples per frequency (default: 25)")

    # Optimization trajectory (default: 75 iters, snapshot every step → ~60% of dataset)
    # Convergence happens fast, so more frequencies with fewer iters is preferred
    # over fewer frequencies with many iters.
    parser.add_argument("--opt-iters", type=int, default=75,
                        help="Optimization steps per frequency (0 = skip, default: 75)")
    parser.add_argument("--opt-lr", type=float, default=0.005,
                        help="Learning rate for optimization (default: 0.005)")
    parser.add_argument("--opt-snapshot", type=int, default=1,
                        help="Snapshot every N optimization steps (default: 1)")
    parser.add_argument("--opt-freqs", type=float, nargs="+", default=None,
                        help="Run optimization only at these frequencies (default: all)")

    parser.add_argument("--seed", type=int, default=42, help="RNG seed")

    args = parser.parse_args()

    # Load base config
    base_config = load_config(args.config)

    if not base_config.cells.enabled:
        print("Error: cells.enabled must be true in the base config.")
        sys.exit(1)

    # Build frequency grid
    if args.f_stars is not None:
        f_stars = sorted(args.f_stars)
    else:
        f_stars = np.arange(args.f_min, args.f_max + 1e-6, args.f_step).tolist()
        # Round to avoid float artifacts
        f_stars = [round(f, 4) for f in f_stars]

    # Output path
    if args.output is not None:
        output_path = args.output
    else:
        output_path = str(Path(base_config.output_dir) / "surrogate_dataset.h5")

    # Build generation config
    gen_config = DatasetGenConfig(
        f_stars=f_stars,
        n_random_per_freq=args.n_random,
        noise_scales=args.noise_scales,
        n_smooth_per_freq=args.n_smooth,
        opt_n_iters=args.opt_iters,
        opt_lr=args.opt_lr,
        opt_snapshot_every=args.opt_snapshot,
        opt_f_stars=args.opt_freqs,
        seed=args.seed,
        output_path=output_path,
    )

    print(f"Base config: {args.config}")
    print(f"Output:      {output_path}")
    print(f"Frequencies: {len(f_stars)} values from {f_stars[0]:.2f} to {f_stars[-1]:.2f}")
    print(f"Random:      {args.n_random}/freq, noise scales={args.noise_scales}")
    print(f"Smooth:      {args.n_smooth}/freq (low-freq sinusoidal + noise)")
    if args.opt_iters > 0:
        print(f"Opt traj:    {args.opt_iters} iters, lr={args.opt_lr}, "
              f"snapshot every {args.opt_snapshot}")
        if args.opt_freqs:
            print(f"  opt freqs: {args.opt_freqs}")
    else:
        print(f"Opt traj:    DISABLED")
    print()

    run_dataset_generation(base_config, gen_config)


if __name__ == "__main__":
    main()
