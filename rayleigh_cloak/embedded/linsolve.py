"""Direct sparse solve of a linear jax-fem problem without PETSc.

jax-fem assembles the residual and the element Jacobians
(``problem.newton_update``); we build the global CSR matrix ourselves, impose
Dirichlet rows by row elimination (same as ``jax_fem.solver.get_A``) and
factorise with MKL PARDISO (``pypardiso``), falling back to SuperLU.
For a linear problem one Newton step from u = 0 is exact.
"""

from __future__ import annotations

import time

import jax
import jax.flatten_util
import numpy as np
import scipy.sparse as sp


def solve_linear(problem, backend: str = "pardiso") -> tuple[np.ndarray, dict]:
    t0 = time.time()
    fe = problem.fes[0]
    n = problem.num_total_dofs_all_vars
    sol0 = [np.zeros((f.num_total_nodes, f.vec)) for f in problem.fes]
    res_list = problem.newton_update(sol0)
    res = np.asarray(jax.flatten_util.ravel_pytree(res_list)[0], dtype=np.float64)
    A = sp.csr_matrix((np.asarray(problem.V), (problem.I, problem.J)), shape=(n, n))
    t_asm = time.time() - t0

    rows, vals = [], []
    for nodes, comp, v in zip(fe.node_inds_list, fe.vec_inds_list, fe.vals_list):
        rows.append(np.asarray(nodes) * fe.vec + np.asarray(comp))
        vals.append(np.broadcast_to(np.asarray(v, dtype=np.float64), np.shape(nodes)))
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    keep = np.ones(n)
    keep[rows] = 0.0
    A = (sp.diags(keep) @ A + sp.diags(1.0 - keep)).tocsr()
    rhs = -res
    rhs[rows] = vals

    t1 = time.time()
    if backend == "pardiso":
        try:
            import pypardiso
            x = pypardiso.spsolve(A, rhs)
        except ImportError:
            backend = "superlu"
    if backend == "superlu":
        from scipy.sparse.linalg import spsolve
        x = spsolve(A.tocsc(), rhs)
    t_solve = time.time() - t1
    rel_res = float(np.linalg.norm(A @ x - rhs) / max(np.linalg.norm(rhs), 1e-300))
    if not np.isfinite(rel_res) or rel_res > 1e-6:
        raise RuntimeError(f"linear solve inaccurate: relative residual {rel_res:.3e}")
    u = x.reshape(-1, fe.vec)
    info = dict(n_dofs=int(n), nnz=int(A.nnz), t_assemble=t_asm, t_solve=t_solve,
                rel_residual=rel_res, backend=backend, n_dirichlet=int(len(rows)))
    return u, info
