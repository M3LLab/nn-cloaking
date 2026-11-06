"""Step 2 baselines: how much is lost by pixels and by Cauchy symmetry, with no optimisation.

For each radial map, each pixel grid and each sampling method: the ideal tensor in pixels, and
its symmetrised (Cauchy) version. Also the continuous symmetrised cloak, which isolates the cost
of symmetry from the cost of pixels. Same sources, forces, frequency and metrics as Step 1.
Each map and source runs in a fresh subprocess; finished ones are reused.

    .venv/bin/python scripts/circular_pixel_baseline.py --config configs/circular_embedded/step2.yaml
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FORCES = {"vertical": (0.0, 1.0), "horizontal": (1.0, 0.0)}


def worker(cfg_json, grids, methods, src, out_dir):
    from rayleigh_cloak.embedded import memguard
    from rayleigh_cloak.embedded.adjoint import assemble, force_rhs, make_backend
    from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
    from rayleigh_cloak.embedded.linsolve import solve_linear_directions
    from rayleigh_cloak.embedded.mesh import build_mesh
    from rayleigh_cloak.embedded.pixels import PixelProblem, PolarPixelGrid, sample_ideal, symmetrize_many
    from rayleigh_cloak.embedded.problem import build_problem
    from rayleigh_cloak.embedded.sampling import PointEvaluator, exterior_grid, rel_l2, rel_l2_mag

    cfg = EmbeddedCloakConfig.model_validate_json(cfg_json)
    P = exterior_grid(cfg)
    full = build_mesh(cfg, msh_path=os.path.join(out_dir, f"_{src}.msh"), verbose=False)
    sub, _ = full.without_disk()
    dirs = list(FORCES.values())

    def fields(uu, mesh):
        ev = PointEvaluator(mesh.points, mesh.cells, P)
        V, H = (ev(u[:, :2] + 1j * u[:, 2:]) for u in uu)
        return {"vertical": V, "horizontal": H, "both": (V + H) / np.sqrt(2.0)}

    pr = build_problem(full, cfg, cloak_on=False)
    A_ref = fields(solve_linear_directions(pr, dirs)[0], full)
    del pr
    rows = []

    def record(label, grid_name, npix, nact, uu):
        F = fields(uu, sub)
        for force in F:
            rows.append(dict(material=label, grid=grid_name, n_pixels=npix, active=nact, source=src, force=force,
                             e=rel_l2_mag(F[force], A_ref[force]), e_c=rel_l2(F[force], A_ref[force])))
        print(f"  {src} {label} {grid_name}: e(vertical) = {rows[-3]['e']:.3e}", flush=True)

    memguard.check_solve(sub.n_cells, sub.ele_type, f"{src} pixel problems", bytes_per_cell=70_000)
    pp = PixelProblem(sub, cfg)
    backend = make_backend()
    asm = None

    def solve_current():
        nonlocal asm
        asm = assemble(pp.problem)
        b = force_rhs(pp.problem, dirs, asm.rows)
        backend.factorize(asm.A)
        x = backend.solve(b).reshape(asm.n, len(dirs))
        res = np.linalg.norm(asm.A @ x - b, axis=0) / np.linalg.norm(b, axis=0)
        if res.max() > 1e-6:
            raise RuntimeError(f"inaccurate solve: {res}")
        return [x[:, k].reshape(-1, 4) for k in range(len(dirs))]

    record("hole (B)", "-", 0, 0, solve_current())
    pp.set_ideal(symmetrize=False)
    record("ideal continuous (C)", "-", 0, 0, solve_current())
    pp.set_ideal(symmetrize=True)
    record("symmetrised continuous", "-", 0, 0, solve_current())
    for nt, nr in grids:
        grid = PolarPixelGrid.from_config(cfg, nt, nr)
        pg = PixelProblem.__new__(PixelProblem)
        pg.__dict__.update(pp.__dict__)
        pg.grid, pg.qp_pixel = grid, grid.index(pp.xq)
        nact = pg.active_pixels()
        for method in methods:
            C, rho = sample_ideal(cfg, grid, method)
            pg.set_pixels(C, rho)
            record(f"ideal pixels, {method}", f"{nt}x{nr}", grid.n_pixels, nact, solve_current())
            n = C.shape[0]
            pg.set_pixels(symmetrize_many(C.reshape(n, 2, 2, 2, 2)), rho)
            record(f"symmetrised pixels, {method}", f"{nt}x{nr}", grid.n_pixels, nact, solve_current())
    backend.free()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/step2.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from rayleigh_cloak.embedded.config import load_embedded_config
    S = yaml.safe_load(open(args.config))
    base = load_embedded_config(S["base_config"])
    out = os.path.join(S["output_dir"], "baseline")
    os.makedirs(out, exist_ok=True)
    B = S["baseline"]
    ctx = mp.get_context("spawn")
    allrows = []
    for kind in B["maps"]:
        for src, xy in S["sources"].items():
            path = os.path.join(out, f"{kind}_{src}.json")
            if os.path.exists(path) and not args.force:
                allrows += json.load(open(path))
                continue
            cfg = base.with_updates({"cloak.radial_map": kind, "source.x": float(xy[0]), "source.y": float(xy[1]),
                                     "output_dir": out})
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                rows = ex.submit(worker, cfg.model_dump_json(), B["grids"], B["methods"], src, out).result()
            for r in rows:
                r["map"] = kind
            json.dump(rows, open(path, "w"), indent=1)
            allrows += rows
    json.dump(allrows, open(os.path.join(out, "baseline.json"), "w"), indent=1)
    L = ["# Step 2 baselines (f* = 1, eps/a = 0.01)", "",
         "`e`: L2 error of ‖u‖ against case A. `e_c`: complex error. Values: worst over the six source and force cases.", "",
         "| map | material | grid | active pixels | e, worst | e_c, worst |", "|---|---|---|---|---|---|"]
    keys = sorted({(r["map"], r["material"], r["grid"], r["active"]) for r in allrows}, key=lambda k: (k[0], k[2], k[1]))
    for m, mat, g, a in keys:
        q = [r for r in allrows if (r["map"], r["material"], r["grid"]) == (m, mat, g)]
        L.append(f"| {m} | {mat} | {g} | {a or '-'} | {max(r['e'] for r in q):.3e} | {max(r['e_c'] for r in q):.3e} |")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
