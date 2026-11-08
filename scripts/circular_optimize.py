"""Neural-field optimisation of the pixelated cloak, using JAX-FEM and an adjoint gradient.

    .venv/bin/python scripts/circular_optimize.py --run NAME
    .venv/bin/python scripts/circular_optimize.py --run NAME --evaluate-only

Runs are defined under ``runs`` in configs/circular_embedded/step2.yaml; the defaults come from
``optimize``. Results are in <output_dir>/optimize/<NAME>/. A run is checkpointed, resumes
automatically, and can be extended by raising ``steps``, so its folder can be moved to another machine.

The best design is the one with the lowest validation error on sources not in the loss. Test sources
take no part in training or selection. The final report covers all sources at the training
frequencies and at ``eval_f_star``.
"""
from __future__ import annotations

import argparse
import json
import math
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


def config_for(base, f_star, kind, out, O):
    f = f_star * base.material.cs / base.cloak.b
    lam = base.material.cs / f
    h = min(O.get("h_max", 0.09375), lam / O.get("min_elems_per_wavelength", 10))
    return base.with_updates({"source.frequency_hz": float(f), "mesh.elems_per_wavelength": float(lam / h),
                              "cloak.radial_map": kind, "output_dir": out})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/step2.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    from rayleigh_cloak.embedded.config import load_embedded_config
    from rayleigh_cloak.embedded.memguard import InsufficientMemory, release_memory
    from rayleigh_cloak.embedded.neural_field import make_neural_field
    from rayleigh_cloak.embedded.optimize_pixels import (
        BandCase, FrequencyCase, band_loss_and_grad, band_validation, loss_and_grad,
    )
    from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many
    from rayleigh_cloak.optimize import AdamState, adam_init, adam_update

    S = yaml.safe_load(open(args.config))
    O = {**S["optimize"], **S["runs"][args.run]}
    base = load_embedded_config(S["base_config"])
    out = os.path.join(S["output_dir"], "optimize", args.run)
    os.makedirs(out, exist_ok=True)
    kind, nt_nr = O["radial_map"], tuple(O["grid"])
    log_path, ckpt_path, best_path = (os.path.join(out, n) for n in ("history.jsonl", "checkpoint.pkl", "best_design.npz"))
    sources = {g: [tuple(p) for p in S["source_groups"][g]] for g in ("train", "val", "test")}
    O.setdefault("f_star", [])

    def make_case(f, srcs, name, loss="complex"):
        cfg = config_for(base, f, kind, out, O)
        return FrequencyCase(cfg, PolarPixelGrid.from_config(cfg, *nt_nr), srcs, loss=loss,
                             train_fraction=O["train_fraction"], backend=O["backend"],
                             msh_path=os.path.join(out, f"_{name}.msh"))

    def final_report():
        d = np.load(best_path)
        rows = []
        for f in sorted(set(O["f_star"]) | set(O.get("eval_f_star", []))):
            try:
                c = make_case(f, sources, f"eval{f:g}")
                for r in c.report(jnp.asarray(d["C"]), jnp.asarray(d["rho"])):
                    r["f_star"] = f
                    rows.append(r)
                del c
            except InsufficientMemory as e:
                print(f"  SKIPPED report at f*={f}: {e}", flush=True)
            release_memory()
        json.dump(rows, open(os.path.join(out, "evaluation.json"), "w"), indent=1)
        for f in sorted({r["f_star"] for r in rows}):
            for g in ("train", "val", "test"):
                q = [r["e"] for r in rows if r["f_star"] == f and r["group"] == g]
                qc = [r["e_c"] for r in rows if r["f_star"] == f and r["group"] == g]
                print(f"  f*={f:g} {g:5s} e median {np.median(q):.3e} worst {max(q):.3e} | e_c worst {max(qc):.3e}", flush=True)

    if args.evaluate_only:
        final_report()
        return

    pool = O.get("f_pool")
    cfg0 = config_for(base, max(pool) if pool else O["f_star"][0], kind, out, O)
    grid = PolarPixelGrid.from_config(cfg0, *nt_nr)
    C0, rho0 = sample_ideal(cfg0, grid, "average")
    net = O["network"]
    theta, field = make_neural_field(grid, symmetrize_many(C0), rho0, cfg0.material.mu, **net)
    state = adam_init(theta)
    start, best = 0, np.inf
    if os.path.exists(ckpt_path):
        ck = pickle.load(open(ckpt_path, "rb"))
        theta, start, best = jax.tree.map(jnp.asarray, ck["theta"]), ck["step"], ck["best"]
        state = AdamState(m=jax.tree.map(jnp.asarray, ck["m"]), v=jax.tree.map(jnp.asarray, ck["v"]), t=ck["t"])
        if os.path.exists(best_path):     # the best design can be newer than the last checkpoint
            best = min(best, float(np.load(best_path)["score"]))
        print(f"resuming at step {start}, best validation {best:.4e}", flush=True)
    elif O.get("init_from"):
        ck = pickle.load(open(os.path.join(S["output_dir"], "optimize", O["init_from"], "checkpoint.pkl"), "rb"))
        theta = jax.tree.map(jnp.asarray, ck["theta"])
        state = adam_init(theta)
        print(f"starting from {O['init_from']} at its step {ck['step']}", flush=True)
    if pool:
        case = BandCase(cfg0, grid, {g: sources[g] for g in ("train", "val")}, pool, loss=O["loss"],
                        train_fraction=O["train_fraction"], backend=O["backend"], msh_path=os.path.join(out, "_band.msh"))
        cases = [case]
    else:
        cases = [make_case(f, {g: sources[g] for g in ("train", "val")}, f"train{f:g}", loss=O["loss"]) for f in O["f_star"]]
    print(f"pixels {grid.n_pixels}, active {cases[0].pp.active_pixels()}, frequencies {pool or O['f_star']}, "
          f"train {len(sources['train'])} val {len(sources['val'])} test {len(sources['test'])} sources, "
          f"elements {[c.n_cells for c in cases]}", flush=True)

    steps, lr0, lr1 = O["steps"], O["lr"], O.get("lr_end", O["lr"])
    with open(log_path, "a") as log:
        for step in range(start, steps + 1):
            t0 = time.time()
            if pool:
                # frequencies drawn per step, reproducibly, so a resumed run continues the same sequence
                picks = np.random.default_rng(1000 + step).choice(pool, O["freqs_per_step"], replace=False)
                J, H, g, (C, rho) = band_loss_and_grad(theta, field, cases[0], picks)
                is_val = step % O["val_every"] == 0 or step == steps or step == start   # validate right after a resume
                if is_val:
                    ve = band_validation(cases[0], C, rho)
            else:
                J, H, g, (C, rho), m = loss_and_grad(theta, field, cases)
                ve, is_val = m["val"], True
            if not np.isfinite(J):
                raise RuntimeError(f"non-finite loss at step {step}")
            score = float(np.sqrt(np.mean(ve[:, 1] ** 2)))    # rms complex error on validation sources
            if is_val and score < best:
                best = score
                tmp = best_path + ".tmp.npz"
                np.savez(tmp, C=np.asarray(C), rho=np.asarray(rho), step=step, score=best)
                os.replace(tmp, best_path)
            lr = lr1 + 0.5 * (lr0 - lr1) * (1 + math.cos(math.pi * min(step / max(steps, 1), 1.0)))
            rec = dict(step=step, train=float(J), heldout=float(H), val_rms_ec=score, val_median_e=float(np.median(ve[:, 0])),
                       val_worst_e=float(ve[:, 0].max()), lr=lr, seconds=time.time() - t0,
                       timing={k: round(v, 3) for k, v in cases[0].timing.items()})
            for c in cases:
                c.timing = {}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step % O["print_every"] == 0 or step == steps:
                print(f"step {step:5d}  train e_c {np.sqrt(J):.4e}  val e_c {score:.4e}  val e median {rec['val_median_e']:.3e} "
                      f"worst {rec['val_worst_e']:.3e}  lr {lr:.1e}  {rec['seconds']:.1f}s", flush=True)
            if step == steps:
                break
            upd, state = adam_update(g, state, lr=lr)
            theta = jax.tree.map(lambda a, b: a + b, theta, upd)
            if (step + 1) % O["checkpoint_every"] == 0:
                tmp = ckpt_path + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(dict(theta=jax.tree.map(np.asarray, theta), m=jax.tree.map(np.asarray, state.m),
                                     v=jax.tree.map(np.asarray, state.v), t=state.t, step=step + 1, best=best), fh)
                os.replace(tmp, ckpt_path)
    del cases
    release_memory()
    print(f"best validation rms e_c {best:.4e}; report:", flush=True)
    final_report()


if __name__ == "__main__":
    main()
