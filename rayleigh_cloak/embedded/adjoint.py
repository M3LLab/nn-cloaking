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
    """Factorise once, then solve many right-hand sides (columns of b)."""

    def factorize(self, A: sp.csr_matrix) -> None:
        raise NotImplementedError

    def factorize_csr(self, indptr, indices, data, n) -> None:
        """Factorise from CSR arrays; ``data`` may be a JAX array (device or host)."""
        self.factorize(sp.csr_matrix((np.asarray(data), indices, indptr), shape=(n, n)))

    def residual(self, x, b) -> np.ndarray:
        """Relative residual per column of the last factorised system."""
        r = self.A @ x - b
        return np.linalg.norm(r, axis=0) / np.maximum(np.linalg.norm(b, axis=0), 1e-300)

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


class CuDSSBackend(LinearBackend):
    """GPU direct solver (NVIDIA cuDSS through nvmath-python), following rayleigh_cloak_3d/cudss_solver.py.

    One plan (reordering, symbolic factorisation) for the fixed pattern; each step only uploads the values and
    refactorises. Right-hand sides are solved column by column with the same factor. Not testable on a machine
    without a GPU: run scripts/gpu_selfcheck.py on the server first.
    """

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.solver = None

    def factorize_csr(self, indptr, indices, data, n):
        cp = self.cp
        import cupyx.scipy.sparse as csp
        try:
            d = cp.from_dlpack(data)                          # zero-copy from a JAX device array
        except Exception:
            d = cp.asarray(np.asarray(data))
        if self.solver is None:
            from nvmath.sparse.advanced import DirectSolver, DirectSolverOptions
            from rayleigh_cloak_3d.cudss_solver import _MT_LIB
            self.A = csp.csr_matrix((d.copy(), cp.asarray(indices), cp.asarray(indptr)), shape=(n, n))
            self.b = cp.zeros(n, dtype=self.A.data.dtype)
            self.solver = DirectSolver(self.A, self.b, options=DirectSolverOptions(multithreading_lib=_MT_LIB))
            self.solver.plan()
        else:
            self.A.data[:] = d
        self.solver.factorize()

    def factorize(self, A):
        self.factorize_csr(A.indptr, A.indices, A.data, A.shape[0])

    def solve(self, b):
        cp = self.cp
        b = np.asarray(b)
        cols = b.reshape(b.shape[0], -1)
        out = np.empty_like(cols)
        for k in range(cols.shape[1]):
            self.b[:] = cp.asarray(cols[:, k])
            out[:, k] = cp.asnumpy(self.solver.solve())
        return out.reshape(b.shape)

    def residual(self, x, b):
        cp = self.cp
        X, B = cp.asarray(x), cp.asarray(b)
        r = self.A @ X - B
        return cp.asnumpy(cp.linalg.norm(r, axis=0) / cp.maximum(cp.linalg.norm(B, axis=0), 1e-300))

    def free(self):
        pass          # keep the plan: the pattern is fixed for the whole run


def make_backend(name: str = "pardiso") -> LinearBackend:
    if name == "pardiso":
        return PardisoBackend()
    if name == "cudss":
        return CuDSSBackend()
    raise ValueError(f"unknown linear backend {name!r}; choose pardiso or cudss")


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
    if g.ndim == 2:
        s = s[:, None]
    rhs = s * g
    rhs[rows] = 0.0
    lam = s * backend.solve(rhs).reshape(rhs.shape)
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


@jax.jit
def _sensitivity_batch(grads, vals, jxw, omega, U, L):
    """Sum over right-hand sides k of the sensitivities, for U, L of shape (k, nc, nn, 4)."""
    gu = jnp.einsum("xcan,cqad->xcqnd", U, grads)
    gl = jnp.einsum("xcan,cqad->xcqnd", L, grads)
    uq = jnp.einsum("xcan,qa->xcqn", U, vals)
    lq = jnp.einsum("xcan,qa->xcqn", L, vals)
    dC = (jnp.einsum("cq,xcqij,xcqkl->cqijkl", jxw, gl[..., :2, :], gu[..., :2, :])
          + jnp.einsum("cq,xcqij,xcqkl->cqijkl", jxw, gl[..., 2:, :], gu[..., 2:, :]))
    drho = -omega ** 2 * jxw * jnp.sum(lq * uq, axis=(0, -1))
    return dC, drho


def material_sensitivity_batch(sd: SensitivityData, U: np.ndarray, L: np.ndarray):
    """``material_sensitivity`` summed over several (u, lam) pairs, U and L of shape (k, n_nodes, 4)."""
    return _sensitivity_batch(sd.grads, sd.vals, sd.jxw, float(sd.omega),
                              jnp.asarray(U[:, sd.cells]), jnp.asarray(L[:, sd.cells]))


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
