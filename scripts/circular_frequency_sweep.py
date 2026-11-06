"""Frequency sweep of the embedded cloak: cases A, B, C for three force directions.

    .venv/bin/python scripts/circular_frequency_sweep.py --config configs/circular_embedded/frequency_sweep.yaml
    ... --f-star 1.0 2.5     # only these frequencies
    ... --plots-only         # rebuild figures and the table from saved results

Saved per frequency and source: metrics in f_<f*>.json, and the complex displacement of
cases A, B, C on a regular grid and on rings in f_<f*>_<source>.npz (complex64; `both` is
rebuilt from vertical and horizontal), so further plots need no new solve.

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


# Peak RSS measured for this solver and grid output on meshes above 150k TRI6 elements: 64-69 kB per element.
# memguard still adds its safety factor and reserve.
SOLVE_BYTES_PER_CELL = 70_000
MAP_SPACING = 0.03
RING_OVER_B = 1.25
RING_POINTS = 720


def field_grids(cfg):
    d = cfg.domain
    xs = np.arange(d.x_min, d.x_max + 1e-9, MAP_SPACING)
    ys = np.arange(d.y_min, d.y_max + 1e-9, MAP_SPACING)
    X, Y = np.meshgrid(xs, ys)
    t = np.linspace(0, 2 * np.pi, RING_POINTS, endpoint=False)
    r = RING_OVER_B * cfg.cloak.b
    ring = np.column_stack([cfg.cloak.xc + r * np.cos(t), cfg.cloak.yc + r * np.sin(t)])
    return xs, ys, np.column_stack([X.ravel(), Y.ravel()]), t, ring


def run_source(cfg, name):
    """Return the metrics of one source position at one frequency; save its fields."""
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
    xs, ys, Pmap, t, Pring = field_grids(cfg)
    store = {"xs": xs, "ys": ys, "ring_theta": t}
    for case, mesh, cloak_on in (("A", full, False), ("B", sub, False), ("C", sub, True)):
        memguard.check_solve(mesh.n_cells, mesh.ele_type, f"{name} case {case}",
                             bytes_per_cell=SOLVE_BYTES_PER_CELL)
        problem = build_problem(mesh, cfg, cloak_on=cloak_on)
        (uV, uH), info = solve_linear_directions(problem, [FORCES["vertical"], FORCES["horizontal"]])
        del problem
        ev = PointEvaluator(mesh.points, mesh.cells, P)
        V = ev(uV[:, :2] + 1j * uV[:, 2:])
        H = ev(uH[:, :2] + 1j * uH[:, 2:])
        fields[case] = {"vertical": V, "horizontal": H, "both": (V + H) / np.sqrt(2.0)}
        em, er = PointEvaluator(mesh.points, mesh.cells, Pmap), PointEvaluator(mesh.points, mesh.cells, Pring)
        for force, u4 in (("vertical", uV), ("horizontal", uH)):
            uc = u4[:, :2] + 1j * u4[:, 2:]
            store[f"{case}_{force}"] = em(uc).reshape(len(ys), len(xs), 2).astype(np.complex64)
            store[f"ring_{case}_{force}"] = er(uc).astype(np.complex64)
        print(f"  {name} {case}: {mesh.n_cells} cells, {info['t_assemble']:.0f}+{info['t_solve']:.0f} s, "
              f"res {info['rel_residual']:.1e}", flush=True)
    rss = memguard.peak_rss_bytes() / 2**30
    np.savez_compressed(os.path.join(cfg.output_dir, f"f_{cfg.source.frequency_hz * cfg.cloak.b / cfg.material.cs:g}_{name}.npz"), **store)
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
            row[f"mean_beyond_{case}"] = float(np.nanmean(m[beyond]))
            row[f"mean_all_{case}"] = float(np.nanmean(m))
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

    fig, axes = plt.subplots(len(srcs), 3, figsize=(16, 4.6 * len(srcs)), sharex=True, squeeze=False)
    for i, src in enumerate(srcs):
        for j, force in enumerate(FORCES):
            ax = axes[i, j]
            for key, lab, sty, col in [("mean_beyond_B", "Hole, no cloak (B)", "s--", "k"),
                                       ("mean_beyond_C", "Regularised ideal cloak (C), ε/a = 0.01", "o-", "tab:red"),
                                       ("mean_beyond_A", "No hole, no cloak (A, reference)", "--", "tab:blue")]:
                f, y = _series(rows, src, force, key)
                if len(f):
                    ax.plot(f, y, sty, color=col, ms=4, lw=1.4, label=lab)
            ax.set_title(f"{SRC_LABELS.get(src, src)}, {LABELS[force]}", fontsize=10)
            ax.grid(alpha=0.3)
            if i == len(srcs) - 1:
                ax.set_xlabel(r"$f^* = b f / c_S$")
            if j == 0:
                ax.set_ylabel("⟨‖u‖⟩ beyond the cloak [m]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Mean total displacement vs frequency", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "frequency_sweep_mean_displacement.png"), dpi=150)
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


def _load_mag(z, case, force):
    """Return ‖u‖ on the map grid and on the ring for one case and force."""
    def get(prefix):
        V, H = z[f"{prefix}{case}_vertical"], z[f"{prefix}{case}_horizontal"]
        if force == "both":
            u = (V + H) / np.sqrt(2.0)
        else:
            u = V if force == "vertical" else H
        return np.sqrt(np.sum(np.abs(u) ** 2, axis=-1))
    return get(""), get("ring_")


def make_field_figures(base, sweep, out_dir):
    """One figure per frequency, source and force, plus overviews across frequency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    c = base.cloak
    names = (("A", "A: no hole, no cloak"), ("B", "B: hole, no cloak"), ("C", "C: cloak, ε/a = 0.01"))
    maps = os.path.join(out_dir, "maps")
    os.makedirs(maps, exist_ok=True)
    over = sweep.get("overview_f_star", [])
    for src, xy in sweep["sources"].items():
        data = {}
        for f in sweep["f_star"]:
            p = os.path.join(out_dir, f"f_{f:g}_{src}.npz")
            if os.path.exists(p):
                data[f] = np.load(p)
        for force in FORCES:
            sheet = [f for f in over if f in data]
            if sheet:
                figO, axO = plt.subplots(len(sheet), 3, figsize=(12.5, 3.7 * len(sheet)), squeeze=False)
            for f, z in data.items():
                ext = (float(z["xs"][0]), float(z["xs"][-1]), float(z["ys"][0]), float(z["ys"][-1]))
                X, Y = np.meshgrid(z["xs"], z["ys"])
                fields = {k: _load_mag(z, k, force) for k, _ in names}
                mA = fields["A"][0]
                far = np.hypot(X - xy[0], Y - xy[1]) > base.metrics.source_exclusion
                lim = float(np.percentile(mA[far & np.isfinite(mA)], 99))
                def draw(ax, k, lab):
                    im = ax.imshow(fields[k][0], extent=ext, origin="lower", cmap="inferno", vmin=0, vmax=lim)
                    ax.add_patch(Circle((c.xc, c.yc), c.a, fill=False, ec="w", lw=0.8))
                    ax.add_patch(Circle((c.xc, c.yc), c.b, fill=False, ec="w", lw=0.8, ls="--"))
                    ax.plot(*xy, marker="*", color="gold", mec="k", mew=0.4, ms=9)
                    ax.set_title(lab, fontsize=9)
                    ax.set_aspect("equal")
                    return im
                fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
                for ax, (k, lab) in zip(axes, names):
                    fig.colorbar(draw(ax, k, lab), ax=ax, shrink=0.8, label="‖u‖ [m]")
                t = np.degrees(z["ring_theta"])
                for k, sty, lab in (("B", "k-", "B"), ("A", "b-", "A"), ("C", "r--", "C")):
                    axes[3].plot(t, fields[k][1], sty, lw=1.2, label=lab)
                axes[3].set_title(f"‖u‖ on the ring r = {RING_OVER_B:g} b", fontsize=9)
                axes[3].set_xlabel("θ [deg]")
                axes[3].legend(fontsize=8)
                fig.suptitle(f"f* = {f:g} ({f * base.material.cs / c.b:.0f} Hz), {SRC_LABELS.get(src, src)}, "
                             f"{LABELS[force]}: total displacement, one colour scale", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(maps, f"f_{f:g}_{src}_{force}.png"), dpi=110)
                plt.close(fig)
                if f in sheet:
                    i = sheet.index(f)
                    for j, (k, lab) in enumerate(names):
                        figO.colorbar(draw(axO[i, j], k, f"f* = {f:g}, {lab}"), ax=axO[i, j], shrink=0.8)
            if sheet:
                figO.suptitle(f"Total displacement ‖u‖ [m], {SRC_LABELS.get(src, src)}, {LABELS[force]}; "
                              "colour scale fixed per row", fontsize=12)
                figO.tight_layout()
                figO.savefig(os.path.join(out_dir, f"overview_{src}_{force}.png"), dpi=110)
                plt.close(figO)
    print(f"  wrote maps in {maps}", flush=True)


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
        make_field_figures(base, sweep, out)


if __name__ == "__main__":
    main()
