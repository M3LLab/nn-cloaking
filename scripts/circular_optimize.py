"""Neural-field optimisation of the pixelated cloak, using JAX-FEM and an adjoint gradient.

    .venv/bin/python scripts/circular_optimize.py --config configs/circular_embedded/step2.yaml
    ... --steps 300             # override the number of steps
    ... --name pilot            # results in <output_dir>/optimize/<name>/
    ... --evaluate-only         # evaluate the saved best design only

The run is checkpointed and resumes automatically, so it can be stopped and moved to another
machine with its folder. Training frequencies, sources and pixel grid come from the YAML. The best
design is chosen by the loss on held-out exterior points, not on the training loss. At the end it is
evaluated on all force directions and sources, including held-out ones.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def config_for(base, f_star, xy, kind, out):
    f = f_star * base.material.cs / base.cloak.b
    lam = base.material.cs / f
    h = min(0.09375, lam / 10.0)
    return base.with_updates({"source.frequency_hz": float(f), "source.x": float(xy[0]), "source.y": float(xy[1]),
                              "mesh.elems_per_wavelength": float(lam / h), "cloak.radial_map": kind, "output_dir": out})


def evaluate(design, S, base, out, kind, grid_nt_nr, f_stars):
    """Errors of the design for every source and force, against case A."""
    from rayleigh_cloak.embedded.optimize_pixels import TrainCase
    from rayleigh_cloak.embedded.pixels import PolarPixelGrid
    from rayleigh_cloak.embedded.sampling import rel_l2, rel_l2_mag
    rows = []
    C, rho = jnp.asarray(design["C"]), jnp.asarray(design["rho"])
    for f_star in f_stars:
        for src, xy in S["sources"].items():
            cfg = config_for(base, f_star, xy, kind, out)
            grid = PolarPixelGrid.from_config(cfg, *grid_nt_nr)
            case = TrainCase(cfg, grid, f"eval_{src}", msh_dir=out)
            x, _ = case.solve(C, rho)
            V, H = (case._point_values(x[:, k].reshape(-1, 4)) for k in range(2))
            for force, U, ref in (("vertical", V, case.ref[0]), ("horizontal", H, case.ref[1]),
                                  ("both", (V + H) / np.sqrt(2), (case.ref[0] + case.ref[1]) / np.sqrt(2))):
                rows.append(dict(f_star=f_star, source=src, force=force, e=rel_l2_mag(U, ref), e_c=rel_l2(U, ref),
                                 active_pixels=case.pp.active_pixels()))
            case.backend.free()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/step2.yaml")
    ap.add_argument("--name", default="run")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    from rayleigh_cloak.embedded.config import load_embedded_config
    from rayleigh_cloak.embedded.neural_field import make_neural_field
    from rayleigh_cloak.embedded.optimize_pixels import TrainCase, loss_and_grad
    from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many
    from rayleigh_cloak.optimize import AdamState, adam_init, adam_update

    S = yaml.safe_load(open(args.config))
    O = S["optimize"]
    steps = args.steps or O["steps"]
    lr = args.lr or O["lr"]
    base = load_embedded_config(S["base_config"])
    out = os.path.join(S["output_dir"], "optimize", args.name)
    os.makedirs(out, exist_ok=True)
    kind, nt_nr = O["radial_map"], tuple(O["grid"])
    log_path, ckpt_path, best_path = (os.path.join(out, n) for n in ("history.jsonl", "checkpoint.pkl", "best_design.npz"))

    if args.evaluate_only:
        rows = evaluate(np.load(best_path), S, base, out, kind, nt_nr, O["f_star"])
        print(json.dumps(rows, indent=1))
        return

    cfg0 = config_for(base, O["f_star"][0], S["sources"][O["train_sources"][0]], kind, out)
    grid = PolarPixelGrid.from_config(cfg0, *nt_nr)
    C0, rho0 = sample_ideal(cfg0, grid, "average")
    C0 = symmetrize_many(C0)
    net = O["network"]
    theta, field = make_neural_field(grid, C0, rho0, cfg0.material.mu, hidden=net["hidden"], layers=net["layers"],
                                     n_theta_modes=net["n_theta_modes"], n_r_modes=net["n_r_modes"], seed=net["seed"],
                                     output_scale=net["output_scale"])
    state = adam_init(theta)
    start, best = 0, np.inf
    if os.path.exists(ckpt_path):
        ck = pickle.load(open(ckpt_path, "rb"))
        theta, start, best = jax.tree.map(jnp.asarray, ck["theta"]), ck["step"], ck["best"]
        state = AdamState(m=jax.tree.map(jnp.asarray, ck["m"]), v=jax.tree.map(jnp.asarray, ck["v"]), t=ck["t"])
        print(f"resuming at step {start}, best held-out loss {best:.4e}", flush=True)

    cases = [TrainCase(config_for(base, f, S["sources"][s], kind, out), grid, f"train_f{f:g}_{s}", loss=O["loss"],
                       train_fraction=O["train_fraction"], backend=O["backend"], msh_dir=out)
             for f in O["f_star"] for s in O["train_sources"]]
    print(f"pixels {grid.n_pixels}, active {cases[0].pp.active_pixels()}; {len(cases)} training case(s); "
          f"{cases[0].n_cells} elements in the first", flush=True)

    with open(log_path, "a") as log:
        for step in range(start, steps + 1):
            t0 = time.time()
            J, H, g, (C, rho) = loss_and_grad(theta, field, cases)
            gnorm = float(np.sqrt(sum(float(jnp.sum(x ** 2)) for x in jax.tree.leaves(g))))
            if not np.isfinite(J):
                raise RuntimeError(f"non-finite loss at step {step}")
            if H < best:
                best = float(H)
                tmp = best_path + ".tmp.npz"
                np.savez(tmp, C=np.asarray(C), rho=np.asarray(rho), step=step, heldout=best)
                os.replace(tmp, best_path)
            rec = dict(step=step, train=float(J), heldout=float(H), grad_norm=gnorm, seconds=time.time() - t0)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:4d}  e_train {np.sqrt(J):.4e}  e_heldout {np.sqrt(H):.4e}  "
                  f"|g| {gnorm:.3e}  {rec['seconds']:.1f}s", flush=True)
            if step == steps:
                break
            upd, state = adam_update(g, state, lr=lr)
            theta = jax.tree.map(lambda a, b: a + b, theta, upd)
            if (step + 1) % O["checkpoint_every"] == 0:
                tmp = ckpt_path + ".tmp"
                pickle.dump(dict(theta=jax.tree.map(np.asarray, theta), m=jax.tree.map(np.asarray, state.m),
                                 v=jax.tree.map(np.asarray, state.v), t=state.t, step=step + 1, best=best), open(tmp, "wb"))
                os.replace(tmp, ckpt_path)
    for c in cases:
        c.backend.free()
    del cases
    rows = evaluate(np.load(best_path), S, base, out, kind, nt_nr, O["f_star"])
    json.dump(rows, open(os.path.join(out, "evaluation.json"), "w"), indent=1)
    print("best design evaluation:")
    for r in rows:
        print(f"  f*={r['f_star']:g} {r['source']:13s} {r['force']:10s} e={r['e']:.3e} e_c={r['e_c']:.3e}")


if __name__ == "__main__":
    main()
