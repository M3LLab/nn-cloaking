"""Figures for the embedded circular cloak (Step 1)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
from rayleigh_cloak.embedded.mesh import (
    REGION_ABSORB, REGION_ANNULUS, REGION_BOX, REGION_DISK, TaggedMesh,
)
from rayleigh_cloak.embedded.problem import make_geometry, xi_profile
from rayleigh_cloak.embedded.sampling import PointEvaluator, exterior_grid
from rayleigh_cloak.materials import C_eff, C_iso, C_radial_polar

DPI = 150


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _circles(ax, cfg, lw=0.9, color="k"):
    c = cfg.cloak
    ax.add_patch(Circle((c.xc, c.yc), c.a, fill=False, lw=lw, ec=color))
    ax.add_patch(Circle((c.xc, c.yc), c.b, fill=False, lw=lw, ec=color, ls="--"))


def _source(ax, cfg, size=40):
    s = cfg.source
    d = np.asarray(s.direction, float)
    d = d / np.linalg.norm(d)
    ax.scatter([s.x], [s.y], marker="*", s=size, c="gold", ec="k", lw=0.5, zorder=5)
    ax.annotate("", xy=(s.x + 0.6 * d[0], s.y + 0.6 * d[1]), xytext=(s.x, s.y),
                arrowprops=dict(arrowstyle="->", color="k", lw=1.2), zorder=6)


def _mesh_edges(mesh: TaggedMesh, sel=None):
    cells = mesh.cells if sel is None else mesh.cells[sel]
    p = mesh.points
    tri = cells[:, :3]
    segs = np.concatenate([p[tri[:, [0, 1]]], p[tri[:, [1, 2]]], p[tri[:, [2, 0]]]])
    return segs


# ── setup / geometry / BC / source ───────────────────────────────────


def plot_setup(cfg: EmbeddedCloakConfig, path: str):
    c, d, s = cfg.cloak, cfg.domain, cfg.source
    X0, X1, Y0, Y1 = cfg.outer_box
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))

    ax = axes[0]
    xs = np.linspace(X0, X1, 400)
    ys = np.linspace(Y0, Y1, 400)
    XX, YY = np.meshgrid(xs, ys)
    xi = np.asarray(jax.vmap(xi_profile(cfg))(jnp.stack([XX.ravel(), YY.ravel()], 1))).reshape(XX.shape)
    im = ax.imshow(xi, extent=(X0, X1, Y0, Y1), origin="lower", cmap="Greys", vmin=0,
                   vmax=max(cfg.absorbing.xi_max, 1e-9) * 1.3)
    fig.colorbar(im, ax=ax, shrink=0.75, label=r"absorbing-layer loss $\xi(\mathbf{x})$")
    ax.add_patch(Rectangle((d.x_min, d.y_min), d.x_max - d.x_min, d.y_max - d.y_min,
                           fill=False, ec="tab:blue", lw=1.2, label="physical box"))
    ax.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec="tab:red", lw=3,
                           label="clamped outer edge  u = 0"))
    ax.add_patch(Circle((c.xc, c.yc), c.b, fc="tab:orange", alpha=0.6, ec="k", lw=0.8,
                        label=f"cloak a<r<b (b={c.b} m)"))
    ax.add_patch(Circle((c.xc, c.yc), c.a, fc="white", ec="k", lw=1.0,
                        label=f"traction-free cavity r<a (a={c.a} m)"))
    _source(ax, cfg, 90)
    ax.scatter([], [], marker="*", c="gold", ec="k", label="Gaussian force (vertical)")
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect("equal")
    ax.set_title(f"Computational domain: absorbing layer L={cfg.absorbing.thickness} m on all sides\n"
                 f"f={s.frequency_hz:g} Hz, $\\lambda_S$={cfg.lambda_s:.2f} m, "
                 f"$\\lambda_P$={cfg.lambda_p:.2f} m", fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    ax = axes[1]
    P = exterior_grid(cfg, spacing=max(cfg.metrics.grid_spacing, 0.15))
    ax.scatter(P[:, 0], P[:, 1], s=1.5, c="tab:green", alpha=0.6,
               label=f"exterior sampling (r > {1 + cfg.metrics.exterior_margin:g} b, "
                     f"|x-xs| > {cfg.metrics.source_exclusion} m)")
    ax.add_patch(Circle((c.xc, c.yc), c.b, fc="tab:orange", alpha=0.6, ec="k", lw=1.2, ls="--"))
    ax.add_patch(Circle((c.xc, c.yc), c.a, fc="white", ec="k", lw=1.6))
    ax.add_patch(Circle((s.x, s.y), cfg.metrics.source_exclusion, fill=False, ec="gray", ls=":"))
    ax.add_patch(Circle((s.x, s.y), 3 * s.sigma, fc="gold", alpha=0.5, ec="none"))
    _source(ax, cfg, 90)
    ax.annotate("traction-free\n" r"$\sigma\cdot n = 0$ at r = a", (c.xc, c.yc), ha="center",
                va="center", fontsize=8)
    ax.annotate("bonded interface r = b\n[u] = 0, [σ·n] = 0",
                (c.xc + c.b * 0.71, c.yc + c.b * 0.71), xytext=(c.xc + 2.0, c.yc + 2.6),
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.annotate(f"source ({s.x}, {s.y})\nσ={s.sigma} m, F0={s.amplitude:g} N/m",
                (s.x, s.y), xytext=(s.x + 0.5, s.y - 1.4), fontsize=8,
                arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(d.x_min, d.x_max); ax.set_ylim(d.y_min, d.y_max); ax.set_aspect("equal")
    ax.set_title("Physical box: BCs, source, metric sampling set", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, markerscale=4)
    _save(fig, path)


def plot_mesh(mesh: TaggedMesh, cfg: EmbeddedCloakConfig, path: str):
    c, s = cfg.cloak, cfg.source
    cols = {REGION_DISK: "tab:gray", REGION_ANNULUS: "tab:orange",
            REGION_BOX: "tab:blue", REGION_ABSORB: "k"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    views = [
        (axes[0, 0], cfg.outer_box, "full mesh"),
        (axes[0, 1], (c.xc - 1.3 * c.b, c.xc + 1.3 * c.b, c.yc - 1.3 * c.b, c.yc + 1.3 * c.b), "cloak"),
        (axes[1, 0], (c.xc + 0.6 * c.a, c.xc + 1.25 * c.a, c.yc - 0.3 * c.a, c.yc + 0.35 * c.a),
         "inner boundary r = a (graded)"),
        (axes[1, 1], (s.x - 0.8, s.x + 0.8, s.y - 0.8, s.y + 0.8), "source"),
    ]
    cent = mesh.points[mesh.cells[:, :3]].mean(axis=1)
    for ax, (x0, x1, y0, y1), title in views:
        pad = 0.05 * (x1 - x0)
        inview = ((cent[:, 0] > x0 - pad) & (cent[:, 0] < x1 + pad)
                  & (cent[:, 1] > y0 - pad) & (cent[:, 1] < y1 + pad))
        for reg, col in cols.items():
            sel = inview & (mesh.region == reg)
            if sel.any():
                ax.add_collection(LineCollection(_mesh_edges(mesh, sel), lw=0.15 if title == "full mesh" else 0.3,
                                                 colors=col))
        _circles(ax, cfg, lw=0.8, color="r")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
    sz = mesh.sizes
    fig.suptitle(f"{mesh.ele_type} mesh: {mesh.n_cells} elements, {mesh.n_nodes} nodes | "
                 f"h_bg={sz.get('h_bg', 0):.3f}  h_cloak={sz.get('h_cloak', 0):.3f}  "
                 f"h(r=a)={sz.get('h_inner', 0):.2e}  h_src={sz.get('h_src', 0):.3f} m\n"
                 "black: absorbing, blue: box, orange: cloak, gray: disk (case A only)", fontsize=10)
    _save(fig, path)


def plot_material(cfg: EmbeddedCloakConfig, path: str):
    """Radial profiles of the polar components and Cartesian maps."""
    c, m = cfg.cloak, cfg.material
    geo = make_geometry(cfg)
    C0 = C_iso(m.lam, m.mu)
    r = np.linspace(c.a * (1 + 1e-6), c.b, 600)
    R, Fr, Ft = (np.asarray(v) for v in geo.radial.stretches(jnp.asarray(r)))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    Cp = np.asarray(jax.vmap(lambda a, b_: C_radial_polar(m.lam, m.mu, a, b_))(Fr, Ft))
    M = m.lam + 2 * m.mu
    for (idx, lab) in [((0, 0, 0, 0), r"$C_{rrrr}/(\lambda+2\mu)$"), ((1, 1, 1, 1), r"$C_{\theta\theta\theta\theta}/(\lambda+2\mu)$"),
                       ((0, 1, 0, 1), r"$C_{r\theta r\theta}/\mu$"), ((1, 0, 1, 0), r"$C_{\theta r\theta r}/\mu$"),
                       ((0, 0, 1, 1), r"$C_{rr\theta\theta}/\lambda$"), ((0, 1, 1, 0), r"$C_{r\theta\theta r}/\mu$")]:
        norm = M if idx in [(0, 0, 0, 0), (1, 1, 1, 1)] else (m.lam if idx == (0, 0, 1, 1) else m.mu)
        sty = {(0, 0, 0, 0): "-", (1, 1, 1, 1): "-", (0, 1, 0, 1): "--", (1, 0, 1, 0): "--",
               (0, 0, 1, 1): "-", (0, 1, 1, 0): ":"}[idx]
        ax.semilogy(r, Cp[:, idx[0], idx[1], idx[2], idx[3]] / norm, sty, lw=2 if sty == "-" else 1.3, label=lab)
    ax.semilogy(r, 1 / (Fr * Ft), "k--", label=r"$\rho'/\rho = 1/J$")
    ax.axvline(c.a, color="gray", lw=0.5); ax.axvline(c.b, color="gray", lw=0.5)
    ax.set_xlabel("r [m]"); ax.legend(fontsize=7)
    ax.text(0.98, 0.5, "pairs coincide:\n$C_{rrrr}/(\\lambda+2\\mu)=C_{\\theta r\\theta r}/\\mu=F_r/F_\\theta$\n"
            "$C_{\\theta\\theta\\theta\\theta}/(\\lambda+2\\mu)=C_{r\\theta r\\theta}/\\mu=F_\\theta/F_r$",
            transform=ax.transAxes, ha="right", fontsize=7)
    ax.set_title(f"polar components ({c.radial_map}, ε={c.eps:g} m, ε/a={c.eps / c.a:g})", fontsize=10)

    n = 241
    xs = np.linspace(c.xc - c.b, c.xc + c.b, n)
    XX, YY = np.meshgrid(xs, xs + (c.yc - c.xc))
    pts = jnp.stack([XX.ravel(), YY.ravel()], 1)
    Cc = np.asarray(jax.vmap(lambda x: C_eff(x, geo, C0))(pts)).reshape(n, n, 2, 2, 2, 2)
    rr = np.hypot(XX - c.xc, YY - c.yc)
    mask = (rr < c.a) | (rr > c.b)
    ax = axes[1]
    v = np.log10(Cc[..., 0, 0, 0, 0] / M)
    v[mask] = np.nan
    im = ax.imshow(v, extent=(xs[0], xs[-1], XX[0, 0] * 0 + YY[0, 0], YY[-1, 0]), origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}(C'_{xxxx}/(\lambda+2\mu))$")
    ax.set_title("Cartesian $C'_{xxxx}$ (FEM frame)", fontsize=10); ax.set_aspect("equal")
    ax = axes[2]
    minor = np.linalg.norm((Cc - np.swapaxes(Cc, 2, 3)).reshape(n, n, -1), axis=-1) / \
        np.linalg.norm(Cc.reshape(n, n, -1), axis=-1)
    minor[mask] = np.nan
    im = ax.imshow(minor, extent=(xs[0], xs[-1], YY[0, 0], YY[-1, 0]), origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\|C'_{ijkl}-C'_{jikl}\|/\|C'\|$")
    ax.set_title("minor-symmetry violation (Cosserat part)", fontsize=10); ax.set_aspect("equal")
    for ax in axes[1:]:
        _circles(ax, cfg, lw=0.6)
    _save(fig, path)


# ── fields ───────────────────────────────────────────────────────────


def field_images(cfg: EmbeddedCloakConfig, res: dict, spacing: float = 0.03):
    d = cfg.domain
    xs = np.arange(d.x_min, d.x_max + 1e-9, spacing)
    ys = np.arange(d.y_min, d.y_max + 1e-9, spacing)
    XX, YY = np.meshgrid(xs, ys)
    P = np.column_stack([XX.ravel(), YY.ravel()])
    imgs = {k: PointEvaluator(r.mesh.points, r.mesh.cells, P)(r.u).reshape(len(ys), len(xs), 2)
            for k, r in res.items()}
    return (xs[0], xs[-1], ys[0], ys[-1]), imgs, XX, YY


def _lims(img, XX, YY, cfg, q=99.0):
    s = cfg.source
    far = np.hypot(XX - s.x, YY - s.y) > cfg.metrics.source_exclusion
    v = img[far]
    v = v[np.isfinite(v)]
    return float(np.percentile(np.abs(v), q))


LABELS = {"A": "A: homogeneous (no hole)", "B": "B: traction-free hole, no cloak",
          "C": "C: hole + regularised ideal cloak", "D": "D: virtual problem (hole r<ε)"}


def plot_fields(cfg, res, path, cases=("A", "B", "C")):
    ext, imgs, XX, YY = field_images(cfg, {k: res[k] for k in cases})
    A = imgs["A"]
    comps = [("Re $u_x$", lambda u: u[..., 0].real, "RdBu_r"),
             ("Re $u_y$", lambda u: u[..., 1].real, "RdBu_r"),
             ("|u|", lambda u: np.sqrt(np.abs(u[..., 0]) ** 2 + np.abs(u[..., 1]) ** 2), "inferno")]
    fig, axes = plt.subplots(len(cases), 3, figsize=(16, 5.0 * len(cases)))
    for j, (name, fn, cmap) in enumerate(comps):
        lim = _lims(fn(A), XX, YY, cfg)
        for i, k in enumerate(cases):
            ax = axes[i, j]
            vmin = 0 if cmap == "inferno" else -lim
            im = ax.imshow(fn(imgs[k]), extent=ext, origin="lower", cmap=cmap, vmin=vmin, vmax=lim)
            _circles(ax, cfg, color="k" if cmap != "inferno" else "w")
            _source(ax, cfg)
            ax.set_title(f"{LABELS[k]} — {name}", fontsize=9)
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, shrink=0.8, label="m")
    fig.suptitle("Total displacement, common colour scale per column (set by case A, "
                 "99th pct. outside the source neighbourhood)", fontsize=11)
    _save(fig, path)


def plot_scattered(cfg, res, path):
    ext, imgs, XX, YY = field_images(cfg, {k: res[k] for k in ("A", "B", "C")})
    A = imgs["A"]
    sB, sC = imgs["B"] - A, imgs["C"] - A
    # inside r < b the cloak deliberately bends the field (C - A = O(1) there):
    # the scattered field is only meaningful outside the cloak
    rr = np.hypot(XX - cfg.cloak.xc, YY - cfg.cloak.yc)
    sC = np.where((rr < cfg.cloak.b)[..., None], np.nan, sC)
    rmsA = float(np.sqrt(np.nanmean(np.abs(A[..., 0]) ** 2 + np.abs(A[..., 1]) ** 2)))
    lim = _lims(sB[..., 1].real, XX, YY, cfg)
    limC = _lims(sC[..., 1].real, XX, YY, cfg)
    mag = lambda u: np.log10(np.sqrt(np.abs(u[..., 0]) ** 2 + np.abs(u[..., 1]) ** 2) / rmsA + 1e-30)
    rows = [("B − A", sB, lim), ("C − A, r > b (same scale as B − A)", sC, lim),
            (f"C − A, r > b (own scale, ×{lim / limC:.0f})", sC, limC)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    for i, (name, u, l) in enumerate(rows):
        for j, (cn, fn) in enumerate([("Re $u_x$", lambda v: v[..., 0].real),
                                      ("Re $u_y$", lambda v: v[..., 1].real)]):
            ax = axes[i, j]
            im = ax.imshow(fn(u), extent=ext, origin="lower", cmap="RdBu_r", vmin=-l, vmax=l)
            fig.colorbar(im, ax=ax, shrink=0.8, label="m")
            ax.set_title(f"scattered {name} — {cn}", fontsize=9)
        ax = axes[i, 2]
        im = ax.imshow(mag(u), extent=ext, origin="lower", cmap="viridis", vmin=-5, vmax=0)
        fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\log_{10}(|u_{sc}|/\mathrm{rms}|u_A|)$")
        ax.set_title(f"scattered {name} — magnitude", fontsize=9)
        for ax in axes[i]:
            _circles(ax, cfg); _source(ax, cfg); ax.set_aspect("equal")
    fig.suptitle("Scattered field relative to the homogeneous reference A "
                 "(cloak interior r < b masked for C: the field there is the transformed one)", fontsize=11)
    _save(fig, path)


def plot_lines(cfg, res, path):
    c, s = cfg.cloak, cfg.source
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    for ax, rho in zip(axes[:2], cfg.metrics.ring_radii_over_b[:2]):
        P = np.column_stack([c.xc + rho * c.b * np.cos(t), c.yc + rho * c.b * np.sin(t)])
        for k, sty in [("A", "k-"), ("B", "r-"), ("C", "b--"), ("D", "g:")]:
            if k in res:
                u = res[k].evaluator(P)(res[k].u)
                ax.plot(np.degrees(t), np.sqrt((np.abs(u) ** 2).sum(1)), sty, lw=1.2, label=k)
        ax.set_xlabel("θ [deg]"); ax.set_ylabel("|u| [m]")
        ax.set_title(f"|u| on the ring r = {rho:g} b", fontsize=10); ax.legend(fontsize=8)
    ax = axes[2]
    d = np.array([c.xc - s.x, c.yc - s.y]); d /= np.linalg.norm(d)
    sline = np.linspace(-6, 6, 1200)
    P = np.column_stack([c.xc + sline * d[0], c.yc + sline * d[1]])
    inbox = ((P[:, 0] > cfg.domain.x_min) & (P[:, 0] < cfg.domain.x_max)
             & (P[:, 1] > cfg.domain.y_min) & (P[:, 1] < cfg.domain.y_max))
    P, sline = P[inbox], sline[inbox]
    for k, sty in [("A", "k-"), ("B", "r-"), ("C", "b--")]:
        u = res[k].evaluator(P)(res[k].u)
        ax.plot(sline, u[:, 1].real, sty, lw=1.0, label=f"{k} Re $u_y$")
    for x in (-c.b, -c.a, c.a, c.b):
        ax.axvline(x, color="gray", lw=0.5)
    ax.set_xlabel("signed distance from cloak centre along source→centre line [m]")
    ax.set_title("cut through source and cloak (shadow side > b)", fontsize=10)
    ax.legend(fontsize=8)
    _save(fig, path)


# ── studies ──────────────────────────────────────────────────────────


def plot_eps_study(rows, b_level, a, path):
    eps = np.array([r["eps"] for r in rows]) / a
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, lab, sty in [("C_vs_ref", "C vs A (cloak error)", "bo-"),
                          ("D_vs_ref", "D vs A (intrinsic ε error)", "gs--"),
                          ("C_vs_D", "C vs D (implementation/discretisation)", "r^-"),
                          ("interior", "interior: u_C(x) vs u_D(Ξ⁻¹x)", "mv:")]:
        y = np.array([r.get(key, np.nan) for r in rows])
        ax.loglog(eps, y, sty, label=lab)
    ref = rows[len(rows) // 2]
    e0 = ref["eps"] / a
    ax.loglog(eps, ref["D_vs_ref"] * (eps / e0) ** 2, "k:", lw=0.8, label="∝ ε²")
    ax.axhline(b_level, color="r", lw=0.8, ls="-.", label=f"B (no cloak) = {b_level:.3f}")
    ax.set_xlabel("ε / a"); ax.set_ylabel("normalised complex L2 error (exterior)")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Regularisation convergence")
    _save(fig, path)


def plot_mesh_study(rows, path):
    h = np.array([r["h_bg"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    for key, lab, sty in [("C_vs_ref", "C vs A (same mesh)", "bo-"),
                          ("D_vs_ref", "D vs A (same exterior mesh size)", "gs--"),
                          ("C_vs_D", "C vs D", "r^-"),
                          ("A_vs_analytic", "A vs analytic Green tensor", "k*-"),
                          ("A_vs_finest", "A vs finest-mesh A", "c.-"),
                          ("C_vs_finest", "C vs finest-mesh C", "m.-")]:
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        y[y <= 0] = np.nan          # the finest level vs itself is exactly 0
        if np.isfinite(y).any():
            ax.loglog(h, y, sty, label=lab)
    ax.set_xlabel("background element size h_bg [m]"); ax.set_ylabel("normalised complex L2")
    ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Mesh sensitivity (TRI6)")
    _save(fig, path)
