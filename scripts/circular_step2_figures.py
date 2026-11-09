"""Figures and data for the Step 2 report and paper, from the saved runs.

    .venv/bin/python scripts/circular_step2_figures.py            # all figures
    .venv/bin/python scripts/circular_step2_figures.py --no-fields # skip the field maps (they need solves)

Reads docs/figures/circular_step2/ (run histories, evaluations, best designs, baseline, hole reference) and writes
the figures there. Comparisons between the baselines and the optimised designs use the two Step 1 sources, the only
ones in both; test-source statistics use the four test sources.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

D = os.path.join(ROOT, "docs/figures/circular_step2")
RUNS = ["f1_500", "f1_500_linear", "f1_252", "f1_120", "f1_60", "band_500", "pool_500"]
STEP1 = [[-3.0, -3.0], [-3.5, -2.0]]
LAB = {"f1_500": "f* = 1, 500 px", "f1_500_linear": "f* = 1, 500 px, linear map", "f1_252": "f* = 1, 252 px",
       "f1_120": "f* = 1, 120 px", "f1_60": "f* = 1, 60 px", "band_500": "3 frequencies, 500 px",
       "pool_500": "frequency pool, 500 px"}


def J(*p):
    return json.load(open(os.path.join(D, *p)))


def stats(rows, f, group="test", sources=None):
    q = [r["e"] for r in rows if abs(r["f_star"] - f) < 1e-9 and r["group"] == group
         and (sources is None or r["source"] in sources)]
    return (float(np.median(q)), float(max(q))) if q else (np.nan, np.nan)


def save(fig, name):
    fig.savefig(os.path.join(D, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def fig_frequency(hole):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, k, title in ((axes[0], 0, "median over test sources and forces"), (axes[1], 1, "worst")):
        fs = sorted({r["f_star"] for r in hole})
        ax.plot(fs, [stats(hole, f)[k] for f in fs], "ks--", label="hole, no cloak (B)")
        for r, sty in (("f1_500", "o-"), ("band_500", "^-"), ("pool_500", "D-")):
            R = J("runs", r, "evaluation.json")
            fs = sorted({x["f_star"] for x in R})
            ax.plot(fs, [stats(R, f)[k] for f in fs], sty, label=LAB[r])
        ax.axvspan(0.5, 1.5, color="0.92", zorder=0)
        ax.set_xlabel(r"$f^* = b f / c_S$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$e$, normalised L2 error of $\|\mathbf{u}\|$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Optimised pixel cloaks on unseen test sources against frequency (grey: pool training band)", fontsize=11)
    save(fig, "fig_frequency.png")


def fig_pixels(base):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    n = [60, 120, 252, 500]
    names = {60: "f1_60", 120: "f1_120", 252: "f1_252", 500: "f1_500"}
    grid = {60: "15x4", 120: "24x5", 252: "36x7", 500: "50x10"}
    opt = [stats(J("runs", names[k], "evaluation.json"), 1.0, sources=STEP1)[1] for k in n]
    for mat, sty in (("ideal pixels, centre", "o--"), ("ideal pixels, average", "s--"), ("symmetrised pixels, average", "v:")):
        ax.plot(n, [max(r["e"] for r in base if r["map"] == "quadratic" and r["material"] == mat and r["grid"] == grid[k]) for k in n],
                sty, label=f"{mat} (no optimisation)")
    ax.plot(n, opt, "D-", lw=2, label="optimised Cauchy pixels")
    hole = max(r["e"] for r in base if r["map"] == "quadratic" and r["material"] == "hole (B)")
    ideal = max(r["e"] for r in base if r["map"] == "quadratic" and r["material"] == "ideal continuous (C)")
    ax.axhline(hole, color="k", ls="--", lw=1, label=f"hole, no cloak ({hole:.3f})")
    ax.axhline(ideal, color="tab:green", ls="-.", lw=1, label=f"ideal continuous ({ideal:.1e})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(n); ax.set_xticklabels(n)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("active pixels"); ax.set_ylabel(r"worst $e$ over the two Step 1 sources, $f^*=1$")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    save(fig, "fig_pixels.png")


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for r in ["f1_500", "f1_500_linear", "f1_252", "f1_120", "f1_60"]:
        H = [json.loads(l) for l in open(os.path.join(D, "runs", r, "history.jsonl"))]
        axes[0].plot([h["step"] for h in H], [h["val_rms_ec"] for h in H], label=LAB[r])
    for r in ["band_500", "pool_500"]:
        H = [json.loads(l) for l in open(os.path.join(D, "runs", r, "history.jsonl"))]
        if r == "pool_500":
            H = [h for h in H if h["step"] % 25 == 0]
            off = 600
        else:
            off = 0
        axes[1].plot([h["step"] + off for h in H], [h["val_rms_ec"] for h in H], label=LAB[r] + (" (from step 600 of band)" if off else ""))
    axes[0].set_title("single frequency, f* = 1: validation sources", fontsize=10)
    axes[1].set_title("frequency band: validation sources, all band frequencies", fontsize=10)
    for ax in axes:
        ax.set_xlabel("optimisation step"); ax.set_ylabel(r"rms complex error $e^c$"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig_training.png")


def polar_components(C, grid):
    from rayleigh_cloak.embedded.pixels import PolarPixelGrid  # noqa: F401
    r, t = grid.centres_polar()
    Q = np.stack([np.stack([np.cos(t), np.sin(t)], 1), np.stack([-np.sin(t), np.cos(t)], 1)], axis=2)   # columns e_r, e_t
    return np.einsum("pai,pbj,pck,pdl,pabcd->pijkl", Q, Q, Q, Q, C)


def fig_materials():
    import jax
    jax.config.update("jax_enable_x64", True)
    from rayleigh_cloak.embedded.config import load_embedded_config
    from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many
    cfg = load_embedded_config(os.path.join(ROOT, "configs/circular_embedded/step1.yaml")).with_updates({"cloak.radial_map": "quadratic"})
    grid = PolarPixelGrid.from_config(cfg, 50, 10)
    C0, rho0 = sample_ideal(cfg, grid, "average")
    designs = [("symmetrised initial", np.asarray(symmetrize_many(C0)), np.asarray(rho0))]
    for r in ("f1_500", "pool_500"):
        d = np.load(os.path.join(D, "runs", r, "best_design.npz"))
        designs.append((LAB[r], d["C"], d["rho"]))
    m = cfg.material
    M, mu = m.lam + 2 * m.mu, m.mu
    comps = [((0, 0, 0, 0), M, r"$C_{rrrr}/(\lambda+2\mu)$"), ((1, 1, 1, 1), M, r"$C_{\theta\theta\theta\theta}/(\lambda+2\mu)$"),
             ((0, 1, 0, 1), mu, r"$C_{r\theta r\theta}/\mu$"), ((0, 0, 1, 1), m.lam, r"$C_{rr\theta\theta}/\lambda$"),
             ((0, 0, 0, 1), mu, r"$C_{rrr\theta}/\mu$")]
    fig, axes = plt.subplots(len(designs), 6, figsize=(20, 3.5 * len(designs)), subplot_kw={"projection": "polar"})
    edges_t = np.linspace(0, 2 * np.pi, grid.n_theta + 1)
    edges_r = np.linspace(grid.a, grid.b, grid.n_r + 1)
    for i, (name, C, rho) in enumerate(designs):
        P = polar_components(np.asarray(C), grid)
        cols = [(P[:, a, b_, c, d] / s, lab) for (a, b_, c, d), s, lab in comps] + [(np.asarray(rho) / m.rho, r"$\rho/\rho_0$")]
        for j, (v, lab) in enumerate(cols):
            ax = axes[i, j]
            Z = v.reshape(grid.n_theta, grid.n_r)
            logc = j in (0, 1, 2, 5)
            im = ax.pcolormesh(edges_t, edges_r, (np.log10(np.maximum(Z, 1e-6)) if logc else Z).T, cmap="viridis" if logc else "RdBu_r",
                               shading="flat", **({} if logc else {"vmin": -np.abs(Z).max(), "vmax": np.abs(Z).max()}))
            ax.set_ylim(0, grid.b); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, label=("log10 " if logc else "") + lab)
            if j == 0:
                ax.set_title(name, fontsize=10, loc="left")
    fig.suptitle("Pixel materials in the local polar frame (500 pixels, quadratic map)", fontsize=12)
    save(fig, "fig_materials.png")


def fig_fields():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from circular_optimize import config_for
    import yaml
    from rayleigh_cloak.embedded.adjoint import assemble, force_rhs, make_backend
    from rayleigh_cloak.embedded.config import load_embedded_config
    from rayleigh_cloak.embedded.mesh import build_mesh
    from rayleigh_cloak.embedded.pixels import PixelProblem, PolarPixelGrid
    from rayleigh_cloak.embedded.problem import build_problem
    from rayleigh_cloak.embedded.sampling import PointEvaluator
    S = yaml.safe_load(open(os.path.join(ROOT, "configs/circular_embedded/step2.yaml")))
    base = load_embedded_config(os.path.join(ROOT, S["base_config"]))
    out = os.path.join(ROOT, "output/circular_step2/figures")
    os.makedirs(out, exist_ok=True)
    xs = np.arange(-5, 5.0001, 0.04)
    X, Y = np.meshgrid(xs, xs)
    P = np.column_stack([X.ravel(), Y.ravel()])
    designs = [(r, np.load(os.path.join(D, "runs", r, "best_design.npz"))) for r in ("f1_500", "pool_500")]
    src = (-3.5, -2.0)
    rows = []
    for f in (1.0, 1.5):
        cfg = config_for(base, f, "quadratic", out, S["optimize"]).with_updates({"source.x": src[0], "source.y": src[1]})
        full = build_mesh(cfg, msh_path=os.path.join(out, "_f.msh"), verbose=False)
        sub, _ = full.without_disk()
        be = make_backend()
        fields = {}

        def solve(problem, mesh):
            asm = assemble(problem)
            b = force_rhs(problem, [(0.0, 1.0)], asm.rows)
            be.factorize(asm.A)
            u = be.solve(b).reshape(-1, 4)
            be.free()
            return PointEvaluator(mesh.points, mesh.cells, P)(u[:, :2] + 1j * u[:, 2:]).reshape(len(xs), len(xs), 2)
        fields["A: no hole"] = solve(build_problem(full, cfg, cloak_on=False), full)
        pp = PixelProblem(sub, cfg, PolarPixelGrid.from_config(cfg, 50, 10))
        fields["B: hole"] = solve(pp.problem, sub)
        for r, d in designs:
            pp.set_pixels(jnp.asarray(d["C"]), jnp.asarray(d["rho"]))
            fields[LAB[r]] = solve(pp.problem, sub)
        rows.append((f, fields))
    fig, axes = plt.subplots(2, 4, figsize=(19, 9.4))
    for i, (f, fields) in enumerate(rows):
        mag = {k: np.sqrt((np.abs(v) ** 2).sum(-1)) for k, v in fields.items()}
        far = np.hypot(X - src[0], Y - src[1]) > 0.75
        lim = np.percentile(mag["A: no hole"][far], 99)
        for j, (k, v) in enumerate(mag.items()):
            ax = axes[i, j]
            im = ax.imshow(v, extent=(-5, 5, -5, 5), origin="lower", cmap="inferno", vmin=0, vmax=lim)
            for rad, ls in ((0.5, "-"), (1.5, "--")):
                ax.add_patch(plt.Circle((0, 0), rad, fill=False, ec="w", lw=0.8, ls=ls))
            ax.plot(*src, "*", color="gold", mec="k", ms=9)
            ax.set_title(f"f* = {f:g}: {k}", fontsize=9)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, shrink=0.75, label="‖u‖ [m]")
    fig.suptitle("Total displacement, test source (-3.5, -2), vertical force; one colour scale per row", fontsize=11)
    save(fig, "fig_fields.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fields", action="store_true")
    a = ap.parse_args()
    hole = J("hole_reference.json")
    base = json.load(open(os.path.join(ROOT, "output/circular_step2/baseline/baseline.json"))) \
        if os.path.exists(os.path.join(ROOT, "output/circular_step2/baseline/baseline.json")) else J("baseline.json")
    json.dump(base, open(os.path.join(D, "baseline.json"), "w"), indent=1)
    fig_frequency(hole)
    fig_pixels(base)
    fig_training()
    fig_materials()
    if not a.no_fields:
        fig_fields()


if __name__ == "__main__":
    main()
