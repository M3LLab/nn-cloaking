"""Time one training step, phase by phase, and check a backend against PARDISO. Run this first on the H100.

    .venv/bin/python scripts/benchmark_step.py                       # CPU: fast vs slow assembly
    python scripts/benchmark_step.py --backend cudss --compare       # GPU: cuDSS, checked against PARDISO
    python scripts/benchmark_step.py --scale 1.0 --steps 3           # base mesh

Reports the JAX devices, then per-phase times for each variant. ``--compare`` also solves the same system with
PARDISO and requires the solutions, the adjoint and the gradient to agree, which is the check that cuDSS and the
sign-flip adjoint are correct on that machine.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from rayleigh_cloak.embedded.config import load_embedded_config  # noqa: E402
from rayleigh_cloak.embedded.optimize_pixels import BandCase  # noqa: E402
from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many  # noqa: E402

TRAIN = [(3.234, 1.339), (1.339, 3.234), (-1.339, 3.234), (-3.234, 1.339),
         (-3.234, -1.339), (-1.339, -3.234), (1.339, -3.234), (3.234, -1.339)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="pardiso", choices=["pardiso", "cudss"])
    ap.add_argument("--scale", type=float, default=1.4, help="mesh.scale (1.0 = the training mesh)")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--slow", action="store_true", help="also time the old full assembly")
    ap.add_argument("--compare", action="store_true", help="check the backend against PARDISO")
    a = ap.parse_args()
    print("JAX devices:", jax.devices(), "| x64:", jax.config.jax_enable_x64, flush=True)
    out = os.path.join(ROOT, "output/benchmark")
    os.makedirs(out, exist_ok=True)
    cfg = load_embedded_config(os.path.join(ROOT, "configs/circular_embedded/step1.yaml")).with_updates(
        {"cloak.radial_map": "quadratic", "mesh.scale": a.scale, "source.frequency_hz": 300.0,
         "mesh.elems_per_wavelength": 16.0 * 200 / 300, "output_dir": out})   # f* = 1.5, base element size
    grid = PolarPixelGrid.from_config(cfg, 50, 10)
    C, rho = sample_ideal(cfg, grid, "average")
    C = symmetrize_many(C)
    srcs = {"train": TRAIN, "val": [(2.5, 0.0), (0.0, 4.5)]}
    variants = [("fast", True, a.backend)] + ([("slow", False, a.backend)] if a.slow else [])
    results = {}
    for name, fast, backend in variants:
        t0 = time.perf_counter()
        case = BandCase(cfg, grid, srcs, [1.0, 1.5], backend=backend, fast=fast, msh_path=f"{out}/_m.msh")
        print(f"[{name}/{backend}] setup {time.perf_counter() - t0:.1f}s, {case.n_cells} elements, "
              f"{8 * len(TRAIN) // 4} training right-hand sides", flush=True)
        for s in range(a.steps):
            for f in (1.0, 1.5):
                case.select(f)
                case.timing = {}
                t0 = time.perf_counter()
                J, H, gC, gr, m = case.step(C, rho, groups=("val",))
                total = time.perf_counter() - t0
                ph = "  ".join(f"{k} {v:.2f}" for k, v in case.timing.items())
                print(f"[{name}/{backend}] step {s} f*={f}: total {total:.2f}s | {ph}", flush=True)
        results[name] = (J, np.asarray(gC), np.asarray(gr))
        if a.compare and backend != "pardiso":
            ref = BandCase(cfg, grid, srcs, [1.0, 1.5], backend="pardiso", fast=fast, msh_path=f"{out}/_r.msh")
            ref.select(1.5)
            Jr, _, gCr, grr, _ = ref.step(C, rho, groups=())
            rel = lambda a_, b_: float(np.abs(np.asarray(a_) - np.asarray(b_)).max() / np.abs(np.asarray(b_)).max())
            print(f"[compare] loss {abs(J - Jr) / Jr:.2e}  dJ/dC {rel(gC, gCr):.2e}  dJ/drho {rel(gr, grr):.2e}  "
                  f"(need < 1e-6)", flush=True)
            if max(abs(J - Jr) / Jr, rel(gC, gCr), rel(gr, grr)) > 1e-6:
                raise SystemExit("backend disagrees with PARDISO: do not train with it")
            print("[compare] OK", flush=True)
        del case


if __name__ == "__main__":
    main()
