"""Step 1 driver: continuous regularised ideal cloak, fully embedded circular cavity.

Usage
-----
    .venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1.yaml --study all
    .venv/bin/python scripts/circular_step1.py --config ... --study base        # cases A-D + all field figures
    .venv/bin/python scripts/circular_step1.py --config ... --study eps         # regularisation sweep
    .venv/bin/python scripts/circular_step1.py --config ... --study mesh        # mesh sensitivity
    .venv/bin/python scripts/circular_step1.py --config ... --study absorbing   # absorbing-layer sensitivity
    .venv/bin/python scripts/circular_step1.py --config ... --study map         # linear vs quadratic map
    .venv/bin/python scripts/circular_step1.py --config ... --study summary     # rebuild summary.md only
    .venv/bin/python scripts/circular_step1.py --config ... --polarization horizontal   # one force direction

Every study runs once per entry of ``polarizations`` in the YAML, in
``<output_dir>/<name>/``; ``<output_dir>/summary.md`` compares them.

Every study point runs in a fresh subprocess (memory is returned to the OS
between runs) and is skipped when its ``case_*.npz`` files already exist with
an identical config, so interrupted sweeps resume. ``--force`` re-solves.
Sweep values live in the ``studies`` section of the YAML.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig, load_embedded_config  # noqa: E402


# ── one study point (runs in a subprocess) ──────────────────────────


def _point_worker(cfg_json: str, cases: list[str], out_dir: str, force: bool) -> dict:
    from rayleigh_cloak.embedded.run import compute_metrics, load_cases, run_cases
    cfg = EmbeddedCloakConfig.model_validate_json(cfg_json)
    cfg_path = os.path.join(out_dir, "config.json")
    have = all(os.path.exists(os.path.join(out_dir, f"case_{k}.npz")) for k in cases
               if not (k == "D" and cfg.cloak.eps <= 0))
    same = os.path.exists(cfg_path) and json.load(open(cfg_path)) == json.loads(cfg_json)
    if have and same and not force:
        print(f"=== reuse {out_dir}")
        res = load_cases(out_dir)
    else:
        res = run_cases(cfg, cases=tuple(cases), out_dir=out_dir)
    m = compute_metrics(cfg, res)
    m["info"] = {k: r.info for k, r in res.items()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def run_point(cfg: EmbeddedCloakConfig, cases, out_dir: str, force: bool) -> dict:
    cfg = cfg.with_updates({"output_dir": out_dir})
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        return ex.submit(_point_worker, cfg.model_dump_json(), list(cases), out_dir, force).result()


def _row(m: dict, **extra) -> dict:
    e = m["exterior"]
    row = dict(extra)
    for k in ("B_vs_ref", "C_vs_ref", "D_vs_ref", "C_vs_D", "A_vs_analytic"):
        if k in e:
            row[k] = e[k]
    if "interior_C_vs_D_mapped" in m:
        row["interior"] = m["interior_C_vs_D_mapped"]
        row["mag_interior"] = m["interior_mag_C_vs_D_mapped"]
    for k, v in m.get("exterior_mag", {}).items():
        row[f"mag_{k}"] = v
    for ring, vals in m.get("rings", {}).items():
        for k, v in vals.items():
            row[f"ring{ring}_{k}"] = v
    info = m.get("info", {})
    if "A" in info:
        row["n_cells_A"] = info["A"]["n_cells"]
        row["n_dofs_A"] = info["A"]["n_dofs"]
        row["h_bg"] = info["A"]["mesh_sizes"]["h_bg"]
        row["h_inner"] = info["A"]["mesh_sizes"]["h_inner"]
        row["peak_rss_gib"] = max(v.get("peak_rss_gib", 0) for v in info.values())
    return row


# ── studies ──────────────────────────────────────────────────────────


def study_base(cfg, out, force):
    from rayleigh_cloak.embedded import plots
    from rayleigh_cloak.embedded.mesh import build_mesh
    from rayleigh_cloak.embedded.run import load_cases
    d = os.path.join(out, "base")
    m = run_point(cfg, ["A", "B", "C", "D"], d, force)
    res = load_cases(d)
    fig = os.path.join(out, "figures")
    plots.plot_setup(cfg, os.path.join(fig, "01_setup_bc_source.png"))
    plots.plot_mesh(build_mesh(cfg.with_updates({"output_dir": d}), msh_path=os.path.join(d, "mesh_plot.msh"),
                               verbose=False), cfg, os.path.join(fig, "02_mesh.png"))
    plots.plot_material(cfg, os.path.join(fig, "03_material.png"))
    plots.plot_fields(cfg, res, os.path.join(fig, "04_fields_ABC.png"))
    plots.plot_scattered(cfg, res, os.path.join(fig, "05_scattered.png"))
    plots.plot_lines(cfg, res, os.path.join(fig, "06_lines.png"))
    plots.plot_fields_components(cfg, res, os.path.join(fig, "components", "04_fields_ABC.png"))
    plots.plot_scattered_components(cfg, res, os.path.join(fig, "components", "05_scattered.png"))
    json.dump(_row(m, eps=cfg.cloak.eps), open(os.path.join(out, "base.json"), "w"), indent=2)


def _guarded(fn, *a):
    """Run a study point; a memory-guard refusal is recorded, not fatal."""
    from rayleigh_cloak.embedded.memguard import InsufficientMemory
    try:
        return fn(*a), None
    except InsufficientMemory as e:
        print(f"  SKIPPED (memory guard): {e}")
        return None, str(e)


def study_sweep(cfg, out, name, spec, force):
    rows = []
    for v in spec["values"]:
        d = os.path.join(out, name, f"{spec['key'].split('.')[-1]}_{v:g}")
        m, err = _guarded(run_point, cfg.with_updates({spec["key"]: v}), spec["cases"], d, force)
        key = spec["key"].split(".")[-1]
        rows.append(_row(m, **{key: v}) if m else {key: v, "skipped": "memory guard: run on the big server"})
    return rows


def study_eps(cfg, out, spec, force):
    from rayleigh_cloak.embedded import plots
    rows = study_sweep(cfg, out, "eps", spec, force)
    json.dump(rows, open(os.path.join(out, "eps.json"), "w"), indent=2)
    plots.plot_eps_study(rows, rows[0]["mag_B_vs_ref"], cfg.cloak.a,
                         os.path.join(out, "figures", "07_eps_convergence.png"))


def study_mesh(cfg, out, spec, force):
    from rayleigh_cloak.embedded import plots
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2_mag
    rows = study_sweep(cfg, out, "mesh", spec, force)
    # cross-mesh self-convergence against the finest level, on common sampling points
    done = [i for i, r in enumerate(rows) if "skipped" not in r]
    rows = [rows[i] for i in done]
    vals = [spec["values"][i] for i in done]
    dirs = [os.path.join(out, "mesh", f"scale_{v:g}") for v in vals]
    fin = load_cases(dirs[int(min(range(len(dirs)), key=lambda i: vals[i]))])
    P = exterior_grid(cfg)
    ref = {k: fin[k].evaluator(P)(fin[k].u) for k in ("A", "C")}
    for row, d in zip(rows, dirs):
        res = load_cases(d)
        for k in ("A", "C"):
            row[f"{k}_vs_finest"] = rel_l2_mag(res[k].evaluator(P)(res[k].u), ref[k])
    json.dump(rows, open(os.path.join(out, "mesh.json"), "w"), indent=2)
    plots.plot_mesh_study([r for r in rows], os.path.join(out, "figures", "08_mesh_sensitivity.png"))


def study_variants(cfg, out, name, spec, force, compare_A_to_base=False):
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2_mag
    rows = []
    base_dir = os.path.join(out, "base")
    baseA = load_cases(base_dir).get("A") if compare_A_to_base else None
    P = exterior_grid(cfg)
    for i, upd in enumerate(spec["variants"]):
        d = os.path.join(out, name, f"v{i}")
        m, err = _guarded(run_point, cfg.with_updates(upd), spec["cases"], d, force)
        if m is None:
            rows.append({"variant": json.dumps(upd), "skipped": "memory guard"})
            continue
        row = _row(m, variant=json.dumps(upd))
        if baseA is not None:
            A = load_cases(d)["A"]
            row["A_vs_baseA"] = rel_l2_mag(A.evaluator(P)(A.u), baseA.evaluator(P)(baseA.u))
        rows.append(row)
    json.dump(rows, open(os.path.join(out, f"{name}.json"), "w"), indent=2)


# ── summary ──────────────────────────────────────────────────────────


def _fmt(x):
    return "—" if x is None else (f"{x:.3e}" if isinstance(x, float) else str(x))


def write_summary(cfg, out):
    lines = ["# Step 1 — continuous regularised ideal cloak: results", ""]
    lines.append("Normalised L2 error of the total displacement magnitude ‖u‖ = sqrt(|u_x|² + |u_y|²) on the "
                 f"common exterior sampling grid (spacing {cfg.metrics.grid_spacing} m, "
                 f"r > {1 + cfg.metrics.exterior_margin:g} b, |x − x_s| > {cfg.metrics.source_exclusion} m, "
                 "physical box only). `mag_X_vs_ref` = ‖ ‖u_X‖ − ‖u_A‖ ‖ / ‖ ‖u_A‖ ‖. "
                 "Columns without `mag_` are the complex-displacement error ‖u_X − u_A‖ / ‖u_A‖, "
                 "which also sees phase and direction.\n")
    p = os.path.join(out, "base.json")
    if os.path.exists(p):
        b = json.load(open(p))
        lines += ["## Base case", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in b.items()]
        lines.append("")
    for name, cols in [("eps", ["eps", "mag_B_vs_ref", "mag_C_vs_ref", "mag_D_vs_ref", "mag_C_vs_D", "mag_interior", "C_vs_ref", "h_inner", "n_cells_A"]),
                       ("mesh", ["scale", "h_bg", "n_cells_A", "n_dofs_A", "mag_A_vs_analytic", "mag_B_vs_ref", "mag_C_vs_ref",
                                 "mag_D_vs_ref", "mag_C_vs_D", "A_vs_finest", "C_vs_finest", "C_vs_ref", "peak_rss_gib"]),
                       ("absorbing", ["variant", "mag_A_vs_analytic", "A_vs_baseA", "mag_B_vs_ref", "mag_C_vs_ref", "C_vs_ref", "n_cells_A"]),
                       ("map", ["variant", "mag_C_vs_ref", "mag_D_vs_ref", "mag_C_vs_D", "mag_interior", "C_vs_ref"])]:
        p = os.path.join(out, f"{name}.json")
        if not os.path.exists(p):
            continue
        rows = json.load(open(p))
        lines += [f"## {name} study", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        lines += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows]
        lines.append("")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {os.path.join(out, 'summary.md')}")


# ── polarization comparison ─────────────────────────────────────────


def _mirror_symmetric(cfg) -> bool:
    """True if reflection in y = x maps the whole problem onto itself."""
    c, s, d = cfg.cloak, cfg.source, cfg.domain
    return (c.xc == c.yc and s.x == s.y and d.x_min == d.y_min and d.x_max == d.y_max)


def mirror_errors(cfg, out) -> dict:
    """Relative L2 between the horizontal solve and the mirrored vertical solve."""
    import numpy as np
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2
    V = load_cases(os.path.join(out, "vertical", "base"))
    H = load_cases(os.path.join(out, "horizontal", "base"))
    P = exterior_grid(cfg)
    return {k: rel_l2(H[k].evaluator(P)(H[k].u), V[k].evaluator(P[:, ::-1])(V[k].u)[:, ::-1])
            for k in sorted(V) if k in H}


def write_comparison(cfg, out, names):
    rows = {n: json.load(open(os.path.join(out, n, "base.json")))
            for n in names if os.path.exists(os.path.join(out, n, "base.json"))}
    if not rows:
        return
    names = list(rows)
    L = ["# Step 1: comparison of force directions", "",
         "Exterior normalised L2 error of the total displacement magnitude ‖u‖ on the common "
         "sampling grid (`mag_`); the last rows give the complex-displacement error for reference. "
         "Each direction is a separate solve.", "",
         "## Base case", "", "| metric | " + " | ".join(names) + " |",
         "|---|" + "---|" * len(names)]
    keys = ["mag_B_vs_ref", "mag_C_vs_ref", "mag_D_vs_ref", "mag_C_vs_D", "mag_interior",
            "mag_A_vs_analytic", "ring1.25b_mag_C_vs_ref", "ring2b_mag_C_vs_ref",
            "B_vs_ref", "C_vs_ref", "C_vs_D", "A_vs_analytic"]
    L += [f"| {k} | " + " | ".join(_fmt(rows[n].get(k)) for n in names) + " |" for k in keys]
    eps = {n: json.load(open(os.path.join(out, n, "eps.json")))
           for n in names if os.path.exists(os.path.join(out, n, "eps.json"))}
    if len(eps) == len(names):
        L += ["", "## C vs A against eps (‖u‖ error)", "", "| eps | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
        for i, r in enumerate(eps[names[0]]):
            L.append(f"| {_fmt(r['eps'])} | " + " | ".join(_fmt(eps[n][i].get("mag_C_vs_ref")) for n in names) + " |")
    if {"vertical", "horizontal"} <= set(names) and _mirror_symmetric(cfg):
        L += ["", "## Mirror check", "",
              "The problem is symmetric about y = x, so in the continuum the horizontal field is "
              "the vertical field reflected (x <-> y, u_x <-> u_y). The directions are therefore "
              "not independent tests at this source position; the numbers below measure "
              "the mesh's asymmetry. Complex-displacement error, since this compares the fields themselves.", "", "| case | horizontal vs mirrored vertical |", "|---|---|"]
        L += [f"| {k} | {_fmt(v)} |" for k, v in mirror_errors(cfg, out).items()]
    if {"vertical", "horizontal", "both"} <= set(names):
        L += ["", "## Superposition check", "",
              "The problem is linear, so the `both` solve should equal (u_V + u_H) / sqrt(2). Complex-displacement "
              "error: magnitudes do not superpose.", "",
              "| case | both vs (u_V + u_H) / sqrt(2) |", "|---|---|"]
        L += [f"| {k} | {_fmt(v)} |" for k, v in superposition_errors(cfg, out).items()]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  wrote {os.path.join(out, 'summary.md')}")


def superposition_errors(cfg, out) -> dict:
    import numpy as np
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2
    V, H, B = (load_cases(os.path.join(out, n, "base")) for n in ("vertical", "horizontal", "both"))
    P = exterior_grid(cfg)
    return {k: rel_l2(B[k].evaluator(P)(B[k].u),
                      (V[k].evaluator(P)(V[k].u) + H[k].evaluator(P)(H[k].u)) / np.sqrt(2.0))
            for k in sorted(B) if k in V and k in H}


def run_studies(cfg, studies, todo, force):
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    for s in todo:
        if s == "base":
            study_base(cfg, out, force)
        elif s == "eps":
            study_eps(cfg, out, studies["eps"], force)
        elif s == "mesh":
            study_mesh(cfg, out, studies["mesh"], force)
        elif s == "absorbing":
            study_variants(cfg, out, "absorbing", studies["absorbing"], force, compare_A_to_base=True)
        elif s == "map":
            study_variants(cfg, out, "map", studies["map"], force)
    write_summary(cfg, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/step1.yaml")
    ap.add_argument("--study", default="all",
                    choices=["all", "base", "eps", "mesh", "absorbing", "map", "summary"])
    ap.add_argument("--polarization", default="all",
                    help="a name from `polarizations` in the YAML, or 'all'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_embedded_config(args.config)
    raw = yaml.safe_load(open(args.config)) or {}
    studies = raw.get("studies", {})
    pols = raw.get("polarizations") or {"default": list(cfg.source.direction)}
    if args.polarization == "all":
        selected = list(pols)
    elif args.polarization in pols:
        selected = [args.polarization]
    else:
        ap.error(f"unknown polarization {args.polarization!r}; choose from {list(pols)} or all")
    todo = ["base", "eps", "mesh", "absorbing", "map"] if args.study == "all" else [args.study]
    for name in selected:
        print(f"##### polarization {name}: direction {pols[name]}", flush=True)
        pcfg = cfg.with_updates({"source.direction": pols[name],
                                 "output_dir": os.path.join(cfg.output_dir, name)})
        run_studies(pcfg, studies, todo, args.force)
    write_comparison(cfg, cfg.output_dir, list(pols))


if __name__ == "__main__":
    main()
