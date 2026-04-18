"""Augment an existing surrogate dataset with best-sample frequency sweeps.

For each unique source frequency already present in the HDF5 file, find the
sample with the lowest loss and re-evaluate its (C, rho) at a grid of target
frequencies spanning the dataset's frequency range (same parameters, only
omega changes).  New entries are appended to the same HDF5 file.

Augmented entries use sample_type of the form ``augment_from_f{source:.3f}`` so
subsequent runs can skip them when picking best samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.config import SimulationConfig

from dataset_gen.generate import (
    append_samples,
    build_fixed_context,
    build_freq_context,
    evaluate_loss,
)


def _decode_sample_types(raw) -> np.ndarray:
    return np.array([
        s.decode() if isinstance(s, (bytes, np.bytes_)) else s for s in raw
    ])


def load_best_params_per_freq(h5_path: Path, top_k: int = 1) -> list[dict]:
    """Return the top-k lowest-loss samples for each unique source frequency.

    Samples whose ``sample_type`` begins with ``augment_`` are excluded so the
    function is idempotent when the dataset has already been augmented.
    """
    with h5py.File(h5_path, "r") as f:
        f_stars = f["f_star"][:]
        losses = f["loss"][:]
        sample_types = _decode_sample_types(f["sample_type"][:])
        cell_C_ds = f["cell_C_flat"]
        cell_rho_ds = f["cell_rho"]

        is_augment = np.array([st.startswith("augment_") for st in sample_types])
        valid = ~is_augment

        best = []
        for fs in np.unique(f_stars[valid]):
            mask = (f_stars == fs) & valid
            idxs = np.where(mask)[0]
            order = idxs[np.argsort(losses[idxs])[:top_k]]
            for idx in order:
                best.append({
                    "f_star_source": float(fs),
                    "cell_C_flat": np.asarray(cell_C_ds[idx]),
                    "cell_rho": np.asarray(cell_rho_ds[idx]),
                    "loss_source": float(losses[idx]),
                })
    return best


@dataclass
class AugmentConfig:
    """Controls for dataset augmentation."""

    n_target_freqs: int = 32
    f_min: float | None = None
    f_max: float | None = None
    top_k: int = 1


def run_dataset_augmentation(
    base_config: SimulationConfig,
    h5_path: Path,
    aug_config: AugmentConfig | None = None,
):
    """Augment an HDF5 dataset with frequency sweeps on per-freq best samples."""
    if aug_config is None:
        aug_config = AugmentConfig()

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        existing_f_stars = f["f_star"][:]
    f_min = aug_config.f_min if aug_config.f_min is not None else float(existing_f_stars.min())
    f_max = aug_config.f_max if aug_config.f_max is not None else float(existing_f_stars.max())

    best_samples = load_best_params_per_freq(h5_path, top_k=aug_config.top_k)
    if not best_samples:
        raise RuntimeError(f"No non-augmented samples found in {h5_path}")

    target_f_stars = [
        round(float(f), 4)
        for f in np.linspace(f_min, f_max, aug_config.n_target_freqs)
    ]

    ctx = build_fixed_context(base_config)

    total = len(target_f_stars) * len(best_samples)
    print(f"\n=== Augmentation plan ===")
    print(f"  Source samples:  {len(best_samples)} (top-{aug_config.top_k} per freq)")
    print(f"  Target freqs:    {len(target_f_stars)} "
          f"({target_f_stars[0]:.3f} → {target_f_stars[-1]:.3f})")
    print(f"  Total new evals: {total}")
    print(f"  Output:          {h5_path}\n")

    t_start = time.time()
    n_done = 0

    for fi, f_star in enumerate(target_f_stars):
        print(f"\n{'=' * 60}")
        print(f"  Target freq {fi + 1}/{len(target_f_stars)}: f* = {f_star:.3f}")
        print(f"{'=' * 60}")

        fctx = build_freq_context(ctx, f_star)

        batch = []
        for si, s in enumerate(best_samples):
            params = (jnp.array(s["cell_C_flat"]), jnp.array(s["cell_rho"]))
            t0 = time.time()
            loss = evaluate_loss(fctx, params)
            dt = time.time() - t0
            src = s["f_star_source"]
            print(f"  augment [{si + 1}/{len(best_samples)}] "
                  f"src_f*={src:.2f} target_f*={f_star:.3f} "
                  f"loss={loss:.4e} ({dt:.1f}s)")
            batch.append({
                "cell_C_flat": s["cell_C_flat"],
                "cell_rho": s["cell_rho"],
                "f_star": f_star,
                "loss": loss,
                "sample_type": f"augment_from_f{src:.3f}",
            })
            n_done += 1

        append_samples(h5_path, batch)
        elapsed = time.time() - t_start
        rate = n_done / max(elapsed, 1e-6)
        remain = (total - n_done) / max(rate, 1e-6)
        print(f"  Progress: {n_done}/{total}  "
              f"elapsed={elapsed / 60:.1f}m  eta={remain / 60:.1f}m")

    with h5py.File(h5_path, "r") as f:
        n_total = f["loss"].shape[0]
    print(f"\n{'=' * 60}")
    print(f"  Augmentation complete: {total} new samples (total {n_total})")
    print(f"  Output: {h5_path}")
    print(f"{'=' * 60}")
