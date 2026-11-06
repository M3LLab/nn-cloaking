"""Neural-field optimisation of pixel materials: loss and adjoint gradient.

Training is over a list of ``TrainCase`` objects (one per frequency and source), each with both force
directions. Vertical and horizontal forces share one factorisation. By linearity they span every
in-plane force at that source, so this avoids tuning to a single force direction.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from rayleigh_cloak.embedded import memguard
from rayleigh_cloak.embedded.adjoint import (
    SensitivityData, adjoint_solve, assemble, force_rhs, make_backend, material_sensitivity,
)
from rayleigh_cloak.embedded.linsolve import solve_linear_directions
from rayleigh_cloak.embedded.mesh import build_mesh
from rayleigh_cloak.embedded.pixels import PixelProblem
from rayleigh_cloak.embedded.problem import build_problem
from rayleigh_cloak.embedded.sampling import PointEvaluator, exterior_grid, magnitude

FORCES = {"vertical": (0.0, 1.0), "horizontal": (1.0, 0.0)}
SOLVE_BYTES_PER_CELL = 70_000


def evaluation_matrix(ev: PointEvaluator, n_nodes: int) -> sp.csr_matrix:
    """Sparse matrix P with (P u)[p] = the FE field at point p, for one nodal scalar field."""
    p = np.repeat(np.arange(len(ev.xy)), ev.cells.shape[1])
    a = ev.cells[np.maximum(ev.elem, 0)].ravel()
    return sp.csr_matrix((ev.N.ravel(), (p, a)), shape=(len(ev.xy), n_nodes))


class TrainCase:
    def __init__(self, cfg, grid, name, loss="complex", train_fraction=0.5, seed=0, backend="pardiso",
                 msh_dir=None):
        self.cfg, self.name, self.loss = cfg, name, loss
        msh = f"{msh_dir or cfg.output_dir}/_{name}.msh"
        full = build_mesh(cfg, msh_path=msh, verbose=False)
        sub, _ = full.without_disk()
        memguard.check_solve(sub.n_cells, sub.ele_type, f"{name}", bytes_per_cell=SOLVE_BYTES_PER_CELL)
        self.dirs = list(FORCES.values())
        P = exterior_grid(cfg)
        rng = np.random.default_rng(seed)
        self.train = rng.random(len(P)) < train_fraction
        # reference: case A on the same exterior mesh, so that mesh error cancels
        pr = build_problem(full, cfg, cloak_on=False)
        uu, _ = solve_linear_directions(pr, self.dirs)
        evA = PointEvaluator(full.points, full.cells, P)
        self.ref = [evA(u[:, :2] + 1j * u[:, 2:]) for u in uu]
        del pr, uu, evA, full
        self.pp = PixelProblem(sub, cfg, grid)
        self.sd = SensitivityData(self.pp)
        self.ev_P = evaluation_matrix(PointEvaluator(sub.points, sub.cells, P), sub.n_nodes)
        self.backend = make_backend(backend)
        self.rows = None
        self.rhs = None
        self.n_cells = sub.n_cells
        self.grid_pixels = jnp.asarray(self.pp.qp_pixel)
        self.n_pix = grid.n_pixels

    def _point_values(self, u4):
        re, im = u4[:, :2], u4[:, 2:]
        return self.ev_P @ re + 1j * (self.ev_P @ im)

    def _loss_and_g(self, U, ref, mask):
        """Loss on the masked points, and its gradient wrt the point values (as Re, Im parts)."""
        if self.loss == "complex":
            r = (U - ref) * mask[:, None]
            den = np.sum(np.abs(ref[mask]) ** 2)
            return np.sum(np.abs(r) ** 2) / den, 2 * r.real / den, 2 * r.imag / den
        if self.loss == "magnitude":
            m, mr = magnitude(U), magnitude(ref)
            den = np.sum(mr[mask] ** 2)
            d = (m - mr) * mask
            safe = np.maximum(m, 1e-30)
            return np.sum(d ** 2) / den, (2 * d / safe / den)[:, None] * U.real, (2 * d / safe / den)[:, None] * U.imag
        raise ValueError(f"unknown loss {self.loss!r}")

    def solve(self, C_pix, rho_pix):
        self.pp.set_pixels(C_pix, rho_pix)
        asm = assemble(self.pp.problem)
        if self.rhs is None:
            self.rows = asm.rows
            self.rhs = force_rhs(self.pp.problem, self.dirs, asm.rows)
        self.backend.factorize(asm.A)
        x = self.backend.solve(self.rhs).reshape(asm.n, len(self.dirs))
        res = np.linalg.norm(asm.A @ x - self.rhs, axis=0) / np.linalg.norm(self.rhs, axis=0)
        if not np.all(res < 1e-6):
            raise RuntimeError(f"inaccurate solve in {self.name}: relative residuals {res}")
        return x, asm

    def loss_and_material_grad(self, C_pix, rho_pix, want_grad=True):
        """Mean loss over the force directions on training points, and its gradient wrt pixel values."""
        x, asm = self.solve(C_pix, rho_pix)
        J, heldout = 0.0, 0.0
        gC = jnp.zeros((self.n_pix, 2, 2, 2, 2))
        gr = jnp.zeros(self.n_pix)
        for k in range(len(self.dirs)):
            u4 = x[:, k].reshape(-1, 4)
            U = self._point_values(u4)
            Jk, gre, gim = self._loss_and_g(U, self.ref[k], self.train)
            J += Jk / len(self.dirs)
            heldout += self._loss_and_g(U, self.ref[k], ~self.train)[0] / len(self.dirs)
            if not want_grad:
                continue
            g = np.concatenate([self.ev_P.T @ gre, self.ev_P.T @ gim], axis=1).ravel() / len(self.dirs)
            lam = adjoint_solve(self.backend, g, self.rows).reshape(-1, 4)
            dC, drho = material_sensitivity(self.sd, u4, lam)
            # dJ/d(material) = -lam . dr/d(material)
            gC = gC.at[self.grid_pixels.ravel()].add(-dC.reshape(-1, 2, 2, 2, 2))
            gr = gr.at[self.grid_pixels.ravel()].add(-drho.ravel())
        return J, heldout, gC, gr


def loss_and_grad(theta, field, cases):
    """Mean training loss over ``cases`` and its gradient wrt the network parameters."""
    (C_pix, rho_pix), pull = jax.vjp(field.decode, theta)
    J, H = 0.0, 0.0
    gC, gr = jnp.zeros_like(C_pix), jnp.zeros_like(rho_pix)
    for c in cases:
        j, h, a, b = c.loss_and_material_grad(C_pix, rho_pix)
        J += j / len(cases)
        H += h / len(cases)
        gC = gC + a / len(cases)
        gr = gr + b / len(cases)
    grad = pull((gC, gr))[0]
    return J, H, grad, (C_pix, rho_pix)
