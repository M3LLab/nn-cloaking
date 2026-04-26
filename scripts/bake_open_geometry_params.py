"""Bake the open-geometry shape mask into a "flat" optimized-params npz.

The open-geometry optimizer stores ``(cell_C_flat, cell_rho, shape_logits,
beta, ...)`` per cell. The FEM solver actually evaluates the *blended*
materials::

    m       = sigmoid(beta * smooth(logits))    # per cell
    C_eff   = m^p * C_cell + (1 - m^p) * C0
    rho_eff = m^p * rho_cell + (1 - m^p) * rho0

(see ``rayleigh_cloak.shape_mask.apply_shape_mask``).  Downstream tools like
``scripts/frequency_sweep.py`` only know how to load ``cell_C_flat`` /
``cell_rho`` directly and have no concept of a mask, so they would mis-evaluate
an open-geometry result.

This script applies the mask once and writes a new npz under the keys those
tools expect:: ``cell_C_flat`` and ``cell_rho`` now hold ``C_eff`` /
``rho_eff``.  No other shape changes.

Usage::

    PYTHONPATH=. python scripts/bake_open_geometry_params.py \\
        output/open_geometry/config.yaml \\
        output/open_geometry/optimized_params.npz

    # Custom output path:
    PYTHONPATH=. python scripts/bake_open_geometry_params.py \\
        output/open_geometry/config.yaml \\
        output/open_geometry/optimized_params.npz \\
        -o output/open_geometry/optimized_params_flat.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from rayleigh_cloak import load_config
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.materials import C_iso, _get_converters
from rayleigh_cloak.open_geometry import ShapeOptConfig
from rayleigh_cloak.shape_mask import apply_shape_mask


def bake(config_path: Path, params_path: Path, out_path: Path) -> None:
    cfg = load_config(str(config_path))
    shape_cfg = ShapeOptConfig.from_yaml(str(config_path))
    dp = DerivedParams.from_config(cfg)

    to_flat, _ = _get_converters(cfg.cells.n_C_params)
    C0_flat = jnp.asarray(to_flat(C_iso(dp.lam, dp.mu)))

    d = np.load(params_path)
    required = {"cell_C_flat", "cell_rho", "shape_logits", "beta"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(
            f"{params_path} is missing keys {sorted(missing)}; this script "
            f"expects an open-geometry optimized-params npz."
        )

    C_eff, rho_eff = apply_shape_mask(
        jnp.asarray(d["cell_C_flat"]),
        jnp.asarray(d["cell_rho"]),
        jnp.asarray(d["shape_logits"]),
        C0_flat,
        float(dp.rho0),
        beta=float(d["beta"]),
        simp_p=float(shape_cfg.simp_p),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        cell_C_flat=np.asarray(C_eff),
        cell_rho=np.asarray(rho_eff),
    )

    m = np.asarray(jnp.asarray(d["shape_mask"])) if "shape_mask" in d.files else None
    print(f"Wrote {out_path}")
    print(f"  cell_C_flat: {np.asarray(C_eff).shape}")
    print(f"  cell_rho:    {np.asarray(rho_eff).shape}")
    if m is not None:
        solid = float((m > 0.5).mean())
        grey = float(((m > 0.1) & (m < 0.9)).mean())
        print(f"  mask: solid_frac={solid:.2%}, grey_frac={grey:.2%}, "
              f"beta={float(d['beta']):.2f}, simp_p={shape_cfg.simp_p}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake open-geometry shape mask into a flat optimized-params npz.",
    )
    parser.add_argument("config", help="Path to YAML config used for the run")
    parser.add_argument("params", help="Path to open-geometry optimized_params.npz")
    parser.add_argument(
        "-o", "--out", default=None,
        help="Output npz path (default: <params_dir>/optimized_params_flat.npz)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    params_path = Path(args.params)
    out_path = Path(args.out) if args.out else params_path.with_name("optimized_params_flat.npz")

    bake(config_path, params_path, out_path)


if __name__ == "__main__":
    main()
