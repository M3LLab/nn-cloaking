"""Neural-field optimisation of pixel materials: loss and adjoint gradient.

One ``FrequencyCase`` holds one frequency and all source positions on a single mesh, refined around
every source. The matrix does not depend on the source or force, so one factorisation serves every
right-hand side of that frequency, which makes many sources cheap. Vertical and horizontal forces are both
used: by linearity they span all in-plane forces at each source.

Sources come in groups: ``train`` enters the loss, ``val`` selects the best design, ``test`` is only reported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from rayleigh_cloak.embedded import memguard
from rayleigh_cloak.embedded.adjoint import (
    SensitivityData, adjoint_solve, assemble, make_backend, material_sensitivity,
)
from rayleigh_cloak.embedded.mesh import build_mesh
from rayleigh_cloak.embedded.pixels import PixelProblem
from rayleigh_cloak.embedded.problem import build_problem, layer_coefficients
from rayleigh_cloak.materials import C_iso
from rayleigh_cloak.embedded.sampling import PointEvaluator, magnitude, rel_l2, rel_l2_mag

FORCES = {"vertical": (0.0, 1.0), "horizontal": (1.0, 0.0)}
SOLVE_BYTES_PER_CELL = 70_000


def evaluation_matrix(ev: PointEvaluator, n_nodes: int) -> sp.csr_matrix:
    """Sparse matrix P with (P u)[p] = the FE field at point p, for one nodal scalar field."""
    p = np.repeat(np.arange(len(ev.xy)), ev.cells.shape[1])
    a = ev.cells[np.maximum(ev.elem, 0)].ravel()
    return sp.csr_matrix((ev.N.ravel(), (p, a)), shape=(len(ev.xy), n_nodes))


def candidate_points(cfg) -> np.ndarray:
    """The Step 1 exterior grid before excluding any source neighbourhood."""
    m, c, d = cfg.metrics, cfg.cloak, cfg.domain
    h = m.grid_spacing
    xs = np.arange(d.x_min + m.box_margin, d.x_max - m.box_margin + 1e-12, h)
    ys = np.arange(d.y_min + m.box_margin, d.y_max - m.box_margin + 1e-12, h)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    return P[np.hypot(P[:, 0] - c.xc, P[:, 1] - c.yc) > c.b * (1 + m.exterior_margin)]


class FrequencyCase:
    def __init__(self, cfg, grid, sources: dict, loss="complex", train_fraction=0.5, seed=0,
                 backend="pardiso", msh_path=None):
        """``sources`` maps a group name to a list of (x, y); ``cfg.source`` is only used for its width."""
        self.cfg, self.loss, self.grid = cfg, loss, grid
        self.n_pix = grid.n_pixels
        self.items = []     # (group, source index, xy, force name, direction)
        for group, pts in sources.items():
            for i, xy in enumerate(pts):
                cfg.with_updates({"source.x": float(xy[0]), "source.y": float(xy[1])})   # validates the position
                for name, d in FORCES.items():
                    self.items.append((group, i, tuple(xy), name, d))
        all_xy = [tuple(p) for pts in sources.values() for p in pts]
        full = build_mesh(cfg, msh_path=msh_path or f"{cfg.output_dir}/_case.msh", verbose=False,
                          extra_sources=all_xy)
        sub, _ = full.without_disk()
        self.n_cells = sub.n_cells
        memguard.check_solve(full.n_cells, full.ele_type, f"f={cfg.source.frequency_hz:g} Hz, {len(all_xy)} sources",
                             bytes_per_cell=SOLVE_BYTES_PER_CELL)
        self.P = candidate_points(cfg)
        rng = np.random.default_rng(seed)
        coin = rng.random(len(self.P)) < train_fraction
        self.masks = {}
        for (group, i, xy, name, d) in self.items:
            far = np.hypot(self.P[:, 0] - xy[0], self.P[:, 1] - xy[1]) > cfg.metrics.source_exclusion
            self.masks[(group, i)] = far & coin if group == "train" else far
            self.masks[(group, i, "heldout")] = far & ~coin
            self.masks[(group, i, "all")] = far
        self.backend = make_backend(backend)
        self._rho_scale = 1.0
        self._build_references(full)
        del full
        memguard.release_memory()
        self.pp = PixelProblem(sub, cfg, grid)
        self.sd = SensitivityData(self.pp)
        self.ev = evaluation_matrix(PointEvaluator(sub.points, sub.cells, self.P), sub.n_nodes)
        self.qp_pix = jnp.asarray(self.pp.qp_pixel).ravel()
        self.rhs = None
        self.rows = None
        self.idx = {g: [k for k, it in enumerate(self.items) if it[0] == g] for g in sources}

    def _reference(self, pr, evA, bA=None):
        """Case A at the current frequency, for all right-hand sides, on the same exterior mesh."""
        asmA = assemble(pr)
        if bA is None:
            bA = self._rhs(pr, asmA)
        self.backend.factorize(asmA.A)
        xA = self.backend.solve(bA).reshape(asmA.n, len(self.items))
        self.backend.free()
        return [self._values(evA, xA[:, k].reshape(-1, 4)) for k in range(len(self.items))], bA

    def _build_references(self, full):
        pr = build_problem(full, self.cfg, cloak_on=False)
        evA = evaluation_matrix(PointEvaluator(full.points, full.cells, self.P), full.n_nodes)
        self.ref, _ = self._reference(pr, evA)

    def _rhs(self, problem, asm) -> np.ndarray:
        sol0 = [np.zeros((f.num_total_nodes, f.vec)) for f in problem.fes]
        iv = list(problem.internal_vars)
        b = np.zeros((asm.n, len(self.items)))
        for k, (_, _, xy, _, d) in enumerate(self.items):
            iv[-1] = problem.force_field_at(xy, d)
            res = problem.compute_residual_vars(sol0, iv, problem.internal_vars_surfaces)
            b[:, k] = -np.asarray(jax.flatten_util.ravel_pytree(res)[0], dtype=np.float64)
        b[asm.rows, :] = 0.0
        return b

    @staticmethod
    def _values(E, u4):
        return E @ u4[:, :2] + 1j * (E @ u4[:, 2:])

    def _loss_terms(self, U, ref, mask):
        if self.loss == "complex":
            r = (U - ref) * mask[:, None]
            den = np.sum(np.abs(ref[mask]) ** 2)
            return np.sum(np.abs(r) ** 2) / den, 2 * r.real / den, 2 * r.imag / den
        if self.loss == "magnitude":
            m, mr = magnitude(U), magnitude(ref)
            den = np.sum(mr[mask] ** 2)
            d = (m - mr) * mask
            w = (2 * d / np.maximum(m, 1e-30) / den)[:, None]
            return np.sum(d ** 2) / den, w * U.real, w * U.imag
        raise ValueError(f"unknown loss {self.loss!r}")

    def _factorize(self, C_pix, rho_pix):
        self.pp.set_pixels(C_pix, self._rho_scale * rho_pix)
        asm = assemble(self.pp.problem)
        if self.rhs is None:
            self.rows = asm.rows
            self.rhs = self._rhs(self.pp.problem, asm)
        self.backend.factorize(asm.A)
        return asm

    def _solve_group(self, asm, indices):
        x = self.backend.solve(self.rhs[:, indices]).reshape(asm.n, len(indices))
        res = np.linalg.norm(asm.A @ x - self.rhs[:, indices], axis=0) / np.linalg.norm(self.rhs[:, indices], axis=0)
        if not np.all(res < 1e-6):
            raise RuntimeError(f"inaccurate solve: relative residuals up to {res.max():.2e}")
        return x

    def step(self, C_pix, rho_pix, groups=("val",)):
        """Training loss and gradient wrt pixel values, plus errors for the other ``groups``."""
        asm = self._factorize(C_pix, rho_pix)
        tr = self.idx["train"]
        x = self._solve_group(asm, tr)
        J, H = 0.0, 0.0
        gC = jnp.zeros((self.n_pix, 2, 2, 2, 2))
        gr = jnp.zeros(self.n_pix)
        for j, k in enumerate(tr):
            group, i, *_ = self.items[k]
            u4 = x[:, j].reshape(-1, 4)
            U = self._values(self.ev, u4)
            Jk, gre, gim = self._loss_terms(U, self.ref[k], self.masks[(group, i)])
            J += Jk / len(tr)
            H += self._loss_terms(U, self.ref[k], self.masks[(group, i, "heldout")])[0] / len(tr)
            g = np.concatenate([self.ev.T @ gre, self.ev.T @ gim], axis=1).ravel() / len(tr)
            lam = adjoint_solve(self.backend, g, self.rows).reshape(-1, 4)
            dC, drho = material_sensitivity(self.sd, u4, lam)
            gC = gC.at[self.qp_pix].add(-dC.reshape(-1, 2, 2, 2, 2))
            gr = gr.at[self.qp_pix].add(-drho.ravel())
        metrics = {g: self._group_errors(asm, g) for g in groups if self.idx.get(g)}
        self.backend.free()
        return J, H, gC, gr, metrics

    def _group_errors(self, asm, group):
        """Per-force errors for a source group, with the full exterior mask, as (e, e_c) arrays."""
        ind = self.idx[group]
        x = self._solve_group(asm, ind)
        out = []
        for j, k in enumerate(ind):
            g, i, xy, name, d = self.items[k]
            mask = self.masks[(g, i, "all")]
            U = self._values(self.ev, x[:, j].reshape(-1, 4))
            out.append((rel_l2_mag(U[mask], self.ref[k][mask]), rel_l2(U[mask], self.ref[k][mask])))
        return np.array(out)

    def report(self, C_pix, rho_pix):
        """Errors for every source group and force, including the combined force."""
        asm = self._factorize(C_pix, rho_pix)
        rows = []
        for group, ind in self.idx.items():
            x = self._solve_group(asm, ind)
            U = {k: self._values(self.ev, x[:, j].reshape(-1, 4)) for j, k in enumerate(ind)}
            for k in ind[::2]:
                g, i, xy, *_ = self.items[k]
                mask = self.masks[(g, i, "all")]
                cases = (("vertical", U[k], self.ref[k]), ("horizontal", U[k + 1], self.ref[k + 1]),
                         ("both", (U[k] + U[k + 1]) / np.sqrt(2), (self.ref[k] + self.ref[k + 1]) / np.sqrt(2)))
                for name, u, r in cases:
                    rows.append(dict(group=g, source=list(xy), force=name, e=rel_l2_mag(u[mask], r[mask]),
                                     e_c=rel_l2(u[mask], r[mask])))
        self.backend.free()
        return rows


def frequency_arrays(problem, cfg0, cfg_w):
    """Coefficients of ``problem``, built at cfg0, that give the problem at frequency cfg_w.

    The jitted mass term holds omega0 fixed, so density is scaled by (omega/omega0)^2. The PML
    stretching depends on omega and is recomputed. The force does not depend on frequency.
    """
    m = cfg0.material
    qp = jnp.asarray(problem.physical_quad_points)
    nc, nq, _ = qp.shape
    C = jnp.broadcast_to(C_iso(m.lam, m.mu), (nc, nq, 2, 2, 2, 2))
    rho = jnp.full((nc, nq), m.rho)
    Cc, rc = jax.vmap(jax.vmap(layer_coefficients(cfg_w)))(qp, C, rho)
    s = (cfg_w.omega / cfg0.omega) ** 2
    return [Cc.real, Cc.imag, s * rc.real, s * rc.imag], s


class BandCase(FrequencyCase):
    """All frequencies of a pool on one mesh and one jitted problem, switched cheaply with ``select``.

    Valid while every frequency keeps the mesh of cfg (true below the frequency where the size rule
    follows the wavelength); ``cfg`` must be the highest frequency of the pool.
    """

    def __init__(self, cfg, grid, sources, f_stars, **kw):
        self.f_stars = [float(f) for f in f_stars]
        super().__init__(cfg, grid, sources, **kw)
        self.select(self.f_stars[0])

    def _cfg_at(self, f_star):
        c = self.cfg
        hz = float(f_star * c.material.cs / c.cloak.b)
        return c.with_updates({"source.frequency_hz": hz,        # same element size at every frequency
                               "mesh.elems_per_wavelength": float(c.mesh.elems_per_wavelength * c.source.frequency_hz / hz)})

    def _build_references(self, full):
        pr = build_problem(full, self.cfg, cloak_on=False)
        evA = evaluation_matrix(PointEvaluator(full.points, full.cells, self.P), full.n_nodes)
        self.refs, bA, force = {}, None, pr.internal_vars[-1]
        for f in self.f_stars:
            arrays, _ = frequency_arrays(pr, self.cfg, self._cfg_at(f))
            pr.internal_vars = arrays + [force]
            self.refs[f], bA = self._reference(pr, evA, bA)

    def select(self, f_star):
        """Switch every coefficient and the reference to frequency ``f_star``."""
        f_star = float(f_star)
        cw = self._cfg_at(f_star)
        arrays, s = frequency_arrays(self.pp.problem, self.cfg, cw)
        self.pp.base = arrays + [self.pp.base[-1]]
        self._rho_scale = s
        self.sd.omega = cw.omega
        self.ref = self.refs[f_star]
        self.current = f_star


def band_loss_and_grad(theta, field, case, picks, groups=()):
    """Mean training loss over the frequencies ``picks`` of a BandCase, and its gradient."""
    (C_pix, rho_pix), pull = jax.vjp(field.decode, theta)
    J, H = 0.0, 0.0
    gC, gr = jnp.zeros_like(C_pix), jnp.zeros_like(rho_pix)
    for f in picks:
        case.select(f)
        j, h, a, b, _ = case.step(C_pix, rho_pix, groups)
        J += j / len(picks)
        H += h / len(picks)
        gC = gC + a / len(picks)
        gr = gr + b / len(picks)
    return J, H, pull((gC, gr))[0], (C_pix, rho_pix)


def band_validation(case, C_pix, rho_pix):
    """Per-force errors on validation sources at every frequency of the pool, as an (e, e_c) array."""
    out = []
    for f in case.f_stars:
        case.select(f)
        asm = case._factorize(C_pix, rho_pix)
        out.append(case._group_errors(asm, "val"))
        case.backend.free()
    return np.concatenate(out)


def loss_and_grad(theta, field, cases, groups=("val",)):
    """Mean training loss over the frequency cases and its gradient wrt the network parameters."""
    (C_pix, rho_pix), pull = jax.vjp(field.decode, theta)
    J, H = 0.0, 0.0
    gC, gr = jnp.zeros_like(C_pix), jnp.zeros_like(rho_pix)
    metrics = {}
    for c in cases:
        j, h, a, b, m = c.step(C_pix, rho_pix, groups)
        J += j / len(cases)
        H += h / len(cases)
        gC = gC + a / len(cases)
        gr = gr + b / len(cases)
        for g, v in m.items():
            metrics.setdefault(g, []).append(v)
    grad = pull((gC, gr))[0]
    return J, H, grad, (C_pix, rho_pix), {g: np.concatenate(v) for g, v in metrics.items()}
