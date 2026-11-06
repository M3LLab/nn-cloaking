"""Forward and adjoint solves for pixel design, with one factorisation.

The system matrix is A = K with Dirichlet rows eliminated. The bilinear form is complex
symmetric, so in the real [Re, Im] splitting K^T = S K S with S = diag(1, 1, -1, -1) per node.
Hence the adjoint K^T lam = g on free DOFs is lam = S A^{-1} S g with g = 0 on Dirichlet rows, and
needs no transpose factorisation. The solver sits behind ``LinearBackend`` so that a GPU sparse
solver can be added on the large server without touching the rest.

Design sensitivities use the jax-fem weak form: the term lam . dr/dC is linear in C, and
``material_sensitivity`` evaluates it from jax-fem's own shape data on annulus elements only.
``residual_vjp`` computes the same quantity by reverse-mode differentiation of
``problem.compute_residual_vars`` and is used to test it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


class LinearBackend:
    """Factorise once, then solve many right-hand sides."""

    def factorize(self, A: sp.csr_matrix) -> None:
        raise NotImplementedError

    def solve(self, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def free(self) -> None:
        pass


class PardisoBackend(LinearBackend):
    def __init__(self):
        from pypardiso.pardiso_wrapper import PyPardisoSolver
        self.solver = PyPardisoSolver()
        self.A = None

    def factorize(self, A):
        self.A = A.tocsr()
        self.A.sort_indices()
        self.solver.factorize(self.A)

    def solve(self, b):
        return np.asarray(self.solver.solve(self.A, b))

    def free(self):
        self.solver.free_memory(everything=True)


def make_backend(name: str = "pardiso") -> LinearBackend:
    if name == "pardiso":
        return PardisoBackend()
    raise ValueError(f"unknown linear backend {name!r}; add it in embedded/adjoint.py")


@dataclass
class Assembled:
    A: sp.csr_matrix
    rows: np.ndarray
    n: int


def assemble(problem) -> Assembled:
    """Assemble the matrix at the current material, with Dirichlet rows eliminated."""
    fe = problem.fes[0]
    n = problem.num_total_dofs_all_vars
    sol0 = [np.zeros((f.num_total_nodes, f.vec)) for f in problem.fes]
    problem.newton_update(sol0)
    K = sp.csr_matrix((np.asarray(problem.V), (problem.I, problem.J)), shape=(n, n))
    rows = np.concatenate([np.asarray(a) * fe.vec + np.asarray(c)
                           for a, c in zip(fe.node_inds_list, fe.vec_inds_list)])
    keep = np.ones(n)
    keep[rows] = 0.0
    return Assembled((sp.diags(keep) @ K + sp.diags(1.0 - keep)).tocsr(), rows, n)


def force_rhs(problem, directions, rows) -> np.ndarray:
    """Right-hand sides per force direction. They depend only on the force, so compute once."""
    n = problem.num_total_dofs_all_vars
    sol0 = [np.zeros((f.num_total_nodes, f.vec)) for f in problem.fes]
    saved = problem.internal_vars[-1]
    b = np.zeros((n, len(directions)))
    iv = list(problem.internal_vars)
    for k, d in enumerate(directions):
        iv[-1] = problem.force_field(d)
        res = problem.compute_residual_vars(sol0, iv, problem.internal_vars_surfaces)
        b[:, k] = -np.asarray(jax.flatten_util.ravel_pytree(res)[0], dtype=np.float64)
    problem.internal_vars[-1] = saved
    b[rows, :] = 0.0
    return b


def imag_signs(n: int, vec: int = 4) -> np.ndarray:
    """Diagonal of S: +1 on real DOFs, -1 on imaginary DOFs."""
    return np.where(np.arange(n) % vec >= vec // 2, -1.0, 1.0)


def adjoint_solve(backend: LinearBackend, g: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Solve K^T lam = g on free DOFs using the factorisation of A; lam = 0 on Dirichlet DOFs."""
    s = imag_signs(g.shape[0])
    rhs = s * g
    rhs[rows] = 0.0
    lam = s * backend.solve(rhs)
    lam[rows] = 0.0
    return lam


class SensitivityData:
    """jax-fem shape data of the annulus elements, for fast evaluation of lam . dr/d(material)."""

    def __init__(self, pp):
        fe = pp.problem.fes[0]
        self.cells = np.asarray(fe.cells)[pp.sel]
        self.grads = jnp.asarray(np.asarray(fe.shape_grads)[pp.sel])   # (nc, nq, nn, 2)
        self.vals = jnp.asarray(np.asarray(fe.shape_vals))             # (nq, nn)
        self.jxw = jnp.asarray(np.asarray(fe.JxW)[pp.sel])             # (nc, nq)
        self.omega = pp.cfg.omega


def material_sensitivity(sd: SensitivityData, u: np.ndarray, lam: np.ndarray):
    """Return (dC, drho) at quadrature points: the derivative of lam . r with respect to the
    real annulus stiffness and density, r being the residual at displacement u.

    u, lam have shape (n_nodes, 4), ordered [Re x, Re y, Im x, Im y].
    """
    ue = jnp.asarray(u[sd.cells])                     # (nc, nn, 4)
    le = jnp.asarray(lam[sd.cells])
    gu = jnp.einsum("can,cqad->cqnd", ue, sd.grads)   # (nc, nq, 4, 2)
    gl = jnp.einsum("can,cqad->cqnd", le, sd.grads)
    uq = jnp.einsum("can,qa->cqn", ue, sd.vals)
    lq = jnp.einsum("can,qa->cqn", le, sd.vals)
    w = sd.jxw
    dC = (jnp.einsum("cq,cqij,cqkl->cqijkl", w, gl[:, :, :2], gu[:, :, :2])
          + jnp.einsum("cq,cqij,cqkl->cqijkl", w, gl[:, :, 2:], gu[:, :, 2:]))
    drho = -sd.omega ** 2 * w * jnp.sum(lq * uq, axis=-1)
    return dC, drho


def residual_vjp(pp, u: np.ndarray, lam: np.ndarray):
    """Same as ``material_sensitivity``, by differentiating jax-fem's residual (slower; for tests)."""
    problem = pp.problem
    iv = list(problem.internal_vars)
    sol = [jnp.asarray(u)]
    lam_flat = jnp.asarray(lam).ravel()

    def scalar(C_q, rho_q):
        v = list(iv)
        v[0] = v[0].at[pp.sel].set(C_q)
        v[2] = v[2].at[pp.sel].set(rho_q)
        res = problem.compute_residual_vars(sol, v, problem.internal_vars_surfaces)
        return jnp.dot(jax.flatten_util.ravel_pytree(res)[0], lam_flat)

    return jax.grad(scalar, argnums=(0, 1))(iv[0][pp.sel], iv[2][pp.sel])
