"""Re-homogenise an existing cell-designs directory with the hi-fi static
homogeniser, writing a NEW directory (the source is never modified).

Why
---
Cell-design directories made before 2026-06-17 (``cell_designs``,
``cell_designs_diffusion`` under ``output/cell20_phys_flat4_materialreg4``)
store ``pred_flat4`` from the legacy TRI3 / 1 element-per-pixel jax-fem
homogeniser, which over-predicts thin-cell stiffness. Their matched-homogenised
sweeps (e.g. the paper's 0.967) therefore are not on the same footing as the
``ca_squared_2m`` era runs. This script re-scores the SAME pixel canvases with
the dataset homogeniser (``compute_stiffness_hifi`` + ``_ortho_params_from_C``,
static, TRI6, N=100 -> 2 el/px), so ``frequency_sweep_validated.py
--cell-designs <out>`` evaluates them like-for-like.

Output layout (per cell): ``canvas.npy`` (copied byte-for-byte) and
``weights.npz`` with

  target_flat4, target_rho          copied from the source
  pred_flat4, pred_rho              hi-fi values  [C11, C22, C12, C66], rho
  pred_flat4_source, pred_rho_source  the source's stored values (for diffing)
  homog                             "hifi_TRI6_N100" (string)

Usage
-----
    PYTHONPATH=. python scripts/rehomogenise_cell_designs.py \
        output/cell20_phys_flat4_materialreg4/cell_designs_diffusion \
        -o output/cell20_phys_flat4_materialreg4/rehomog_v1/cell_designs_diffusion_hifi \
        -j 2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np


def _homog(canvas: np.ndarray, N: int, ele_type: str) -> np.ndarray:
    # Imported inside the worker so the spawn pool does not need the parent's state.
    from dataset.stiffness.calc_fem_hifi import compute_stiffness_hifi
    from dataset.cellular_chiral.bulk_stiffness import _ortho_params_from_C

    C_eff, rho_eff, _vf = compute_stiffness_hifi(
        np.asarray(canvas, dtype=np.uint8), N=N, ele_type=ele_type)
    c11, c22, c12, c66 = _ortho_params_from_C(C_eff)
    return np.array([c11, c22, c12, c66, rho_eff], dtype=np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="source cell-designs dir (cell_XXX/canvas.npy + weights.npz)")
    p.add_argument("-o", "--out", type=Path, required=True, help="NEW output dir (must not exist)")
    p.add_argument("--mesh-N", type=int, default=100)
    p.add_argument("--ele-type", default="TRI6")
    p.add_argument("-j", "--workers", type=int, default=1)
    args = p.parse_args()

    if args.out.exists():
        raise SystemExit(f"refusing to write into existing {args.out}")
    cells = sorted(d for d in args.src.glob("cell_*") if (d / "canvas.npy").exists())
    if not cells:
        raise SystemExit(f"no cell_*/canvas.npy under {args.src}")
    canvases = [np.load(d / "canvas.npy") for d in cells]
    print(f"{len(cells)} cells from {args.src}; homog=hifi {args.ele_type} N={args.mesh_N}, "
          f"workers={args.workers}", flush=True)

    t0 = time.time()
    fn = partial(_homog, N=args.mesh_N, ele_type=args.ele_type)
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            res = pool.map(fn, canvases, chunksize=1)
    else:
        res = []
        for i, c in enumerate(canvases):
            res.append(fn(c))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(cells)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"homogenised {len(res)} cells in {time.time() - t0:.0f}s", flush=True)

    args.out.mkdir(parents=True)
    rel = []
    for d, r in zip(cells, res):
        w = np.load(d / "weights.npz")
        od = args.out / d.name
        od.mkdir()
        shutil.copy2(d / "canvas.npy", od / "canvas.npy")
        np.savez(
            od / "weights.npz",
            target_flat4=np.asarray(w["target_flat4"]), target_rho=np.asarray(w["target_rho"]),
            pred_flat4=r[:4], pred_rho=np.float64(r[4]),
            pred_flat4_source=np.asarray(w["pred_flat4"]), pred_rho_source=np.asarray(w["pred_rho"]),
            homog=np.array(f"hifi_{args.ele_type}_N{args.mesh_N}"),
        )
        src = np.append(np.asarray(w["pred_flat4"], float), float(w["pred_rho"]))
        rel.append((r - src) / np.abs(src))
    rel = np.array(rel)
    names = ["C11", "C22", "C12", "C66", "rho"]
    print("hi-fi vs source, relative (median / min / max):")
    for k, n in enumerate(names):
        print(f"  {n:4s} {np.median(rel[:, k]):+.3f}  {rel[:, k].min():+.3f}  {rel[:, k].max():+.3f}")
    np.save(args.out / "rel_change_vs_source.npy", rel)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
