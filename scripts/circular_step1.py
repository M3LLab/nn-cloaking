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
    plots.plot_eps_study(rows, rows[0]["B_vs_ref"], cfg.cloak.a,
                         os.path.join(out, "figures", "07_eps_convergence.png"))


def study_mesh(cfg, out, spec, force):
    from rayleigh_cloak.embedded import plots
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2
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
            row[f"{k}_vs_finest"] = rel_l2(res[k].evaluator(P)(res[k].u), ref[k])
    json.dump(rows, open(os.path.join(out, "mesh.json"), "w"), indent=2)
    plots.plot_mesh_study([r for r in rows], os.path.join(out, "figures", "08_mesh_sensitivity.png"))


def study_variants(cfg, out, name, spec, force, compare_A_to_base=False):
    from rayleigh_cloak.embedded.run import load_cases
    from rayleigh_cloak.embedded.sampling import exterior_grid, rel_l2
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
            row["A_vs_baseA"] = rel_l2(A.evaluator(P)(A.u), baseA.evaluator(P)(baseA.u))
        rows.append(row)
    json.dump(rows, open(os.path.join(out, f"{name}.json"), "w"), indent=2)


# ── summary ──────────────────────────────────────────────────────────


def _fmt(x):
    return "—" if x is None else (f"{x:.3e}" if isinstance(x, float) else str(x))


def write_summary(cfg, out):
    lines = ["# Step 1 — continuous regularised ideal cloak: results", ""]
    lines.append("Normalised complex-displacement L2 error on the common exterior sampling grid "
                 f"(spacing {cfg.metrics.grid_spacing} m, r > {1 + cfg.metrics.exterior_margin:g} b, "
                 f"|x − x_s| > {cfg.metrics.source_exclusion} m, physical box only).  "
                 "`X_vs_ref` = ‖u_X − u_A‖ / ‖u_A‖.\n")
    p = os.path.join(out, "base.json")
    if os.path.exists(p):
        b = json.load(open(p))
        lines += ["## Base case", "", "| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in b.items()]
        lines.append("")
    for name, cols in [("eps", ["eps", "B_vs_ref", "C_vs_ref", "D_vs_ref", "C_vs_D", "interior", "h_inner", "n_cells_A"]),
                       ("mesh", ["scale", "h_bg", "n_cells_A", "n_dofs_A", "A_vs_analytic", "B_vs_ref", "C_vs_ref",
                                 "D_vs_ref", "C_vs_D", "A_vs_finest", "C_vs_finest", "peak_rss_gib"]),
                       ("absorbing", ["variant", "A_vs_analytic", "A_vs_baseA", "B_vs_ref", "C_vs_ref", "n_cells_A"]),
                       ("map", ["variant", "C_vs_ref", "D_vs_ref", "C_vs_D", "interior"])]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/step1.yaml")
    ap.add_argument("--study", default="all",
                    choices=["all", "base", "eps", "mesh", "absorbing", "map", "summary"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_embedded_config(args.config)
    studies = (yaml.safe_load(open(args.config)) or {}).get("studies", {})
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    todo = ["base", "eps", "mesh", "absorbing", "map"] if args.study == "all" else [args.study]
    for s in todo:
        if s == "base":
            study_base(cfg, out, args.force)
        elif s == "eps":
            study_eps(cfg, out, studies["eps"], args.force)
        elif s == "mesh":
            study_mesh(cfg, out, studies["mesh"], args.force)
        elif s == "absorbing":
            study_variants(cfg, out, "absorbing", studies["absorbing"], args.force, compare_A_to_base=True)
        elif s == "map":
            study_variants(cfg, out, "map", studies["map"], args.force)
    write_summary(cfg, out)


if __name__ == "__main__":
    main()
