"""Frequency sweep of the embedded cloak: cases A, B, C for three force directions.

    .venv/bin/python scripts/circular_frequency_sweep.py --config configs/circular_embedded/frequency_sweep.yaml
    ... --f-star 1.0 2.5     # only these frequencies
    ... --plots-only         # rebuild figures and the table from saved results

Each frequency runs in a fresh subprocess, and finished frequencies are reused.
Vertical and horizontal forces share one factorisation per case; the `both` force
is their superposition (u_V + u_H) / sqrt(2), which is exact because the problem is
linear (verified to about 1e-14 by the Step 1 study).

Metric: <|u|> / <|u_ref|> with |u| = sqrt(|u_x|^2 + |u_y|^2) and u_ref from case A,
averaged over the exterior sampling points beyond the cloak as seen from the source,
(x - c).s > b with s the unit vector from source to centre. The whole exterior and
the L2 errors are stored as well.
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

FORCES = {"vertical": (0.0, 1.0), "horizontal": (1.0, 0.0), "both": (1.0, 1.0)}


def frequency_config(base, sweep, f_star, xy, out_dir):
    f = f_star * base.material.cs / base.cloak.b
    lam = base.material.cs / f
    h = min(sweep["h_max"], lam / sweep["min_elems_per_wavelength"])
    return base.with_updates({"source.frequency_hz": float(f), "source.x": float(xy[0]),
                              "source.y": float(xy[1]), "mesh.elems_per_wavelength": float(lam / h),
                              "output_dir": out_dir})


def run_source(cfg, name):
    """Return the metrics of one source position at one frequency."""
    from rayleigh_cloak.embedded import memguard
    from rayleigh_cloak.embedded.analytic import green_gaussian
    from rayleigh_cloak.embedded.linsolve import solve_linear_directions
    from rayleigh_cloak.embedded.mesh import build_mesh
    from rayleigh_cloak.embedded.problem import build_problem
    from rayleigh_cloak.embedded.sampling import PointEvaluator, exterior_grid, magnitude, rel_l2, rel_l2_mag

    c, s = cfg.cloak, cfg.source
    P = exterior_grid(cfg)
    sdir = np.array([c.xc - s.x, c.yc - s.y])
    sdir /= np.linalg.norm(sdir)
    beyond = (P - [c.xc, c.yc]) @ sdir > c.b
    full = build_mesh(cfg, msh_path=os.path.join(cfg.output_dir, f"_{name}.msh"))
    sub, _ = full.without_disk()
    fields, rss = {}, 0.0
    for case, mesh, cloak_on in (("A", full, False), ("B", sub, False), ("C", sub, True)):
        memguard.check_solve(mesh.n_cells, mesh.ele_type, f"{name} case {case}")
        problem = build_problem(mesh, cfg, cloak_on=cloak_on)
        (uV, uH), info = solve_linear_directions(problem, [FORCES["vertical"], FORCES["horizontal"]])
        del problem
        ev = PointEvaluator(mesh.points, mesh.cells, P)
        V = ev(uV[:, :2] + 1j * uV[:, 2:])
        H = ev(uH[:, :2] + 1j * uH[:, 2:])
        fields[case] = {"vertical": V, "horizontal": H, "both": (V + H) / np.sqrt(2.0)}
        print(f"  {name} {case}: {mesh.n_cells} cells, {info['t_assemble']:.0f}+{info['t_solve']:.0f} s, "
              f"res {info['rel_residual']:.1e}", flush=True)
    rss = memguard.peak_rss_bytes() / 2**30
    out = {"n_cells_A": int(full.n_cells), "h_bg": full.sizes["h_bg"], "peak_rss_gib": rss, "forces": {}}
    for force, d in FORCES.items():
        A = fields["A"][force]
        ma = magnitude(A)
        row = {}
        for case in "ABC":
            u = fields[case][force]
            m = magnitude(u)
            row[f"ratio_beyond_{case}"] = float(np.nanmean(m[beyond]) / np.nanmean(ma[beyond]))
            row[f"ratio_all_{case}"] = float(np.nanmean(m) / np.nanmean(ma))
            if case != "A":
                row[f"mag_{case}_vs_A"] = rel_l2_mag(u, A)
                row[f"complex_{case}_vs_A"] = rel_l2(u, A)
        ua = green_gaussian(cfg.with_updates({"source.direction": d}), P)
        row["mag_A_vs_analytic"] = rel_l2_mag(A, ua)
        row["complex_A_vs_analytic"] = rel_l2(A, ua)
        out["forces"][force] = row
    return out


def worker(base_cfg_json, sweep, f_star, out_dir):
    from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
    from rayleigh_cloak.embedded.memguard import InsufficientMemory
    base = EmbeddedCloakConfig.model_validate_json(base_cfg_json)
    res = {"f_star": f_star, "sources": {}}
    print(f"=== f* = {f_star}", flush=True)
    for name, xy in sweep["sources"].items():
        cfg = frequency_config(base, sweep, f_star, xy, out_dir)
        try:
            res["sources"][name] = run_source(cfg, name)
        except InsufficientMemory as e:
            print(f"  SKIPPED {name} at f* = {f_star}: {e}", flush=True)
            res["sources"][name] = {"skipped": str(e)}
    return res


LABELS = {"vertical": "vertical force", "horizontal": "horizontal force", "both": "both directions"}
SRC_LABELS = {"on_diagonal": "source (−3, −3)", "off_diagonal": "source (−3.5, −2)"}


def _series(rows, src, force, key):
    fs, ys = [], []
    for r in rows:
        s = r["sources"].get(src, {})
        if "forces" in s:
            fs.append(r["f_star"])
            ys.append(s["forces"][force][key])
    return np.array(fs), np.array(ys)


def make_figures(rows, sweep, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = list(sweep["sources"])
    fig, axes = plt.subplots(len(srcs), 3, figsize=(16, 4.6 * len(srcs)), sharex=True, sharey=True, squeeze=False)
    for i, src in enumerate(srcs):
        for j, force in enumerate(FORCES):
            ax = axes[i, j]
            for key, lab, sty, col in [("ratio_beyond_B", "Hole, no cloak (B)", "s--", "k"),
                                       ("ratio_beyond_C", "Regularised ideal cloak (C), ε/a = 0.01", "o-", "tab:red"),
                                       ("ratio_beyond_A", "No hole, no cloak (A, reference: 1 by definition)", "--", "tab:blue")]:
                f, y = _series(rows, src, force, key)
                ax.plot(f, y, sty, color=col, ms=4, lw=1.4, label=lab)
            ax.axvline(1.0, color="0.8", lw=0.8, zorder=0)
            ax.set_title(f"{SRC_LABELS.get(src, src)}, {LABELS[force]}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(srcs) - 1:
                ax.set_xlabel(r"$f^* = b f / c_S$")
            if j == 0:
                ax.set_ylabel(r"$\langle|u|\rangle / \langle|u_{\rm ref}|\rangle$ beyond the cloak")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Cloaking performance vs frequency", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frequency_sweep_ratio.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(srcs), 3, figsize=(16, 4.6 * len(srcs)), sharex=True, sharey=True, squeeze=False)
    for i, src in enumerate(srcs):
        for j, force in enumerate(FORCES):
            ax = axes[i, j]
            for key, lab, sty, col in [("mag_B_vs_A", "Hole, no cloak (B) vs A", "s--", "k"),
                                       ("mag_C_vs_A", "Regularised ideal cloak (C) vs A", "o-", "tab:red"),
                                       ("mag_A_vs_analytic", "A vs analytic", "^:", "tab:blue")]:
                f, y = _series(rows, src, force, key)
                ax.semilogy(f, y, sty, color=col, ms=4, lw=1.2, label=lab)
            ax.axvline(1.0, color="0.8", lw=0.8, zorder=0)
            ax.set_title(f"{SRC_LABELS.get(src, src)}, {LABELS[force]}", fontsize=10)
            ax.grid(alpha=0.3, which="both")
            if i == len(srcs) - 1:
                ax.set_xlabel(r"$f^* = b f / c_S$")
            if j == 0:
                ax.set_ylabel("normalised L2 error of ‖u‖ (whole exterior)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Total-displacement error vs frequency", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frequency_sweep_error.png"), dpi=150)
    plt.close(fig)

    L = ["# Frequency sweep: results", "",
         "f* = b f / c_S. `ratio` is <|u|>/<|u_ref|> beyond the cloak; `L2` is the normalised error of ‖u‖ "
         "against A over the whole exterior.", ""]
    for src in srcs:
        for force in FORCES:
            L += [f"## {SRC_LABELS.get(src, src)}, {LABELS[force]}", "",
                  "| f* | ratio B | ratio C | L2 B | L2 C | A vs analytic | elements |", "|---|---|---|---|---|---|---|"]
            for r in rows:
                s = r["sources"].get(src, {})
                if "forces" not in s:
                    L.append(f"| {r['f_star']:g} | skipped (memory) | | | | | |")
                    continue
                q = s["forces"][force]
                L.append(f"| {r['f_star']:g} | {q['ratio_beyond_B']:.4f} | {q['ratio_beyond_C']:.6f} | "
                         f"{q['mag_B_vs_A']:.3e} | {q['mag_C_vs_A']:.3e} | {q['mag_A_vs_analytic']:.3e} | {s['n_cells_A']} |")
            L.append("")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(L))
    print(f"  wrote figures and summary in {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/circular_embedded/frequency_sweep.yaml")
    ap.add_argument("--f-star", type=float, nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    args = ap.parse_args()
    from rayleigh_cloak.embedded.config import load_embedded_config
    sweep = yaml.safe_load(open(args.config))
    base = load_embedded_config(sweep["base_config"])
    out = sweep["output_dir"]
    os.makedirs(out, exist_ok=True)
    fstars = args.f_star or sweep["f_star"]
    path = lambda f: os.path.join(out, f"f_{f:g}.json")
    if not args.plots_only:
        ctx = mp.get_context("spawn")
        for f in fstars:
            if os.path.exists(path(f)) and not args.force:
                print(f"=== reuse f* = {f}", flush=True)
                continue
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                r = ex.submit(worker, base.model_dump_json(), sweep, f, out).result()
            with open(path(f), "w") as fh:
                json.dump(r, fh, indent=1)
    rows = [json.load(open(path(f))) for f in sorted(sweep["f_star"]) if os.path.exists(path(f))]
    if rows:
        make_figures(rows, sweep, out)


if __name__ == "__main__":
    main()
