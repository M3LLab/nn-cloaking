"""Point evaluation of FE fields and the comparison metrics.

``PointEvaluator`` locates arbitrary points in a (possibly curved, TRI6)
triangulation and evaluates the FE interpolant there: the straight-sided
corner triangulation gives a first guess, and a few Newton steps on the
isoparametric map give the exact reference coordinates. Points outside the
mesh (e.g. inside the cavity) evaluate to NaN.
"""

from __future__ import annotations

import numpy as np
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig


def _shape_tri6(xi, eta):
    """TRI6 shape functions + reference gradients, gmsh node order
    (corners 0,1,2; mid-edges 3: 0-1, 4: 1-2, 5: 2-0)."""
    l1, l2, l3 = 1.0 - xi - eta, xi, eta
    N = np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1),
                  4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], axis=-1)
    dxi = np.stack([-(4 * l1 - 1), 4 * l2 - 1, 0 * l3,
                    4 * (l1 - l2), 4 * l3, -4 * l3], axis=-1)
    deta = np.stack([-(4 * l1 - 1), 0 * l2, 4 * l3 - 1,
                     -4 * l2, 4 * l2, 4 * (l1 - l3)], axis=-1)
    return N, dxi, deta


def _shape_tri3(xi, eta):
    N = np.stack([1.0 - xi - eta, xi, eta], axis=-1)
    one = np.ones_like(xi)
    return N, np.stack([-one, one, 0 * one], -1), np.stack([-one, 0 * one, one], -1)


class PointEvaluator:
    def __init__(self, points: np.ndarray, cells: np.ndarray, xy: np.ndarray,
                 tol: float = 1e-7, newton_iters: int = 6):
        self.points = np.asarray(points, dtype=np.float64)
        self.cells = np.asarray(cells)
        self.shape = _shape_tri6 if self.cells.shape[1] == 6 else _shape_tri3
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        self.xy = xy
        tri = Triangulation(self.points[:, 0], self.points[:, 1], self.cells[:, :3])
        elem = tri.get_trifinder()(xy[:, 0], xy[:, 1]).astype(np.int64)
        ref = np.zeros((len(xy), 2))
        ok = elem >= 0
        if ok.any():
            ref[ok], ok_n = self._newton(elem[ok], xy[ok], newton_iters, tol)
            ok[np.nonzero(ok)[0][~ok_n]] = False
        # fallback (curved slivers): try the nearest elements by centroid
        bad = np.nonzero(~ok)[0]
        if len(bad):
            cent = self.points[self.cells[:, :3]].mean(axis=1)
            tree = cKDTree(cent)
            _, cand = tree.query(xy[bad], k=min(12, len(cent)))
            for j in range(cand.shape[1]):
                if len(bad) == 0:
                    break
                e = cand[:len(bad), j] if cand.ndim == 2 else cand
                r, okj = self._newton(e, xy[bad], newton_iters, tol)
                elem[bad[okj]] = e[okj]
                ref[bad[okj]] = r[okj]
                ok[bad[okj]] = True
                keep = ~okj
                bad, cand = bad[keep], cand[keep]
        self.elem = np.where(ok, elem, -1)
        self.ref = ref
        self.found = ok
        N, _, _ = self.shape(ref[:, 0], ref[:, 1])
        self.N = np.where(ok[:, None], N, 0.0)

    def _newton(self, elem, xy, iters, tol):
        X = self.points[self.cells[elem]]                       # (p, nn, 2)
        p0, p1, p2 = X[:, 0], X[:, 1], X[:, 2]
        T = np.stack([p1 - p0, p2 - p0], axis=-1)              # straight-sided guess
        ref = np.linalg.solve(T, (xy - p0)[..., None])[..., 0]
        for _ in range(iters):
            N, dxi, deta = self.shape(ref[:, 0], ref[:, 1])
            x = np.einsum("pn,pnd->pd", N, X)
            Jm = np.stack([np.einsum("pn,pnd->pd", dxi, X),
                           np.einsum("pn,pnd->pd", deta, X)], axis=-1)
            ref = ref - np.linalg.solve(Jm, (x - xy)[..., None])[..., 0]
        N, _, _ = self.shape(ref[:, 0], ref[:, 1])
        resid = np.linalg.norm(np.einsum("pn,pnd->pd", N, X) - xy, axis=1)
        scale = np.linalg.norm(p1 - p0, axis=1)
        inside = ((ref[:, 0] >= -tol) & (ref[:, 1] >= -tol)
                  & (ref.sum(axis=1) <= 1 + tol) & (resid <= 1e-8 * scale + 1e-12))
        return ref, inside

    def __call__(self, u_nodes: np.ndarray) -> np.ndarray:
        """Evaluate a nodal field ``(n_nodes, ...)`` at the points."""
        u_nodes = np.asarray(u_nodes)
        vals = np.einsum("pn,pn...->p...", self.N, u_nodes[self.cells[np.maximum(self.elem, 0)]])
        vals = vals.astype(np.result_type(vals, np.float64), copy=False)
        if np.iscomplexobj(vals):
            vals[~self.found] = np.nan + 1j * np.nan
        else:
            vals[~self.found] = np.nan
        return vals


# ── sampling sets ────────────────────────────────────────────────────


def exterior_grid(cfg: EmbeddedCloakConfig, spacing: float | None = None) -> np.ndarray:
    """Common exterior sampling points: a uniform grid over the physical box
    (minus ``box_margin``), keeping r > b (1 + margin) and |x - xs| > R_src."""
    m, c, d, s = cfg.metrics, cfg.cloak, cfg.domain, cfg.source
    h = m.grid_spacing if spacing is None else spacing
    xs = np.arange(d.x_min + m.box_margin, d.x_max - m.box_margin + 1e-12, h)
    ys = np.arange(d.y_min + m.box_margin, d.y_max - m.box_margin + 1e-12, h)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    r = np.hypot(P[:, 0] - c.xc, P[:, 1] - c.yc)
    rs = np.hypot(P[:, 0] - s.x, P[:, 1] - s.y)
    keep = (r > c.b * (1 + m.exterior_margin)) & (rs > m.source_exclusion)
    return P[keep]


def ring_points(cfg: EmbeddedCloakConfig, radius: float) -> np.ndarray:
    c = cfg.cloak
    t = np.linspace(0, 2 * np.pi, cfg.metrics.ring_points, endpoint=False)
    return np.column_stack([c.xc + radius * np.cos(t), c.yc + radius * np.sin(t)])


def annulus_grid(cfg: EmbeddedCloakConfig, spacing: float, inset: float = 0.0) -> np.ndarray:
    c = cfg.cloak
    xs = np.arange(c.xc - c.b, c.xc + c.b + 1e-12, spacing)
    ys = np.arange(c.yc - c.b, c.yc + c.b + 1e-12, spacing)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    r = np.hypot(P[:, 0] - c.xc, P[:, 1] - c.yc)
    return P[(r > c.a + inset) & (r < c.b - inset)]


def rel_l2(u: np.ndarray, u_ref: np.ndarray) -> float:
    """sqrt(sum |u - u_ref|^2) / sqrt(sum |u_ref|^2) over points and components."""
    u, u_ref = np.asarray(u), np.asarray(u_ref)
    ok = np.all(np.isfinite(u), axis=-1) & np.all(np.isfinite(u_ref), axis=-1)
    num = np.sum(np.abs(u[ok] - u_ref[ok]) ** 2)
    den = np.sum(np.abs(u_ref[ok]) ** 2)
    return float(np.sqrt(num / den))
