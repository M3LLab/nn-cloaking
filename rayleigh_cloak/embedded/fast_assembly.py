"""Incremental assembly for pixel design: only the cloak changes between steps.

The matrix is K = K_rest + K_cloak(material). K_rest (background, PML and boundary rows) is assembled once per
frequency with jax-fem, directly into CSR data with a fixed pattern. Each step only the annulus element
matrices are recomputed, from jax-fem's shape data, and added. ``test_circular_fast_assembly`` checks the result
against a full jax-fem assembly.

All arrays stay in JAX, so on a GPU the CSR data never leaves the device; ``backend.factorize_csr`` takes it from
there (zero-copy to CuPy for cuDSS, a host copy for PARDISO).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


class FastAssembler:
    def __init__(self, pp, sd):
        problem = pp.problem
        fe = problem.fes[0]
        self.n = problem.num_total_dofs_all_vars
        I, J = np.asarray(problem.I, dtype=np.int64), np.asarray(problem.J, dtype=np.int64)
        keys, coo_to_csr = np.unique(I * self.n + J, return_inverse=True)
        self.coo_to_csr = coo_to_csr.astype(np.int64)
        del I, J
        rows, cols = keys // self.n, keys % self.n
        self.nnz = int(keys.size)
        self.indptr = np.searchsorted(rows, np.arange(self.n + 1)).astype(np.int32)
        self.indices = cols.astype(np.int32)
        del keys
        # Dirichlet rows: zero them and put 1 on the diagonal (row elimination, as jax-fem)
        self.bc_rows = np.concatenate([np.asarray(a) * fe.vec + np.asarray(c)
                                       for a, c in zip(fe.node_inds_list, fe.vec_inds_list)])
        bc = np.zeros(self.n, bool)
        bc[self.bc_rows] = True
        row_of = np.repeat(np.arange(self.n), np.diff(self.indptr))
        self.bc_entries = np.nonzero(bc[row_of])[0]
        self.bc_diag = np.nonzero(bc[row_of] & (self.indices == row_of))[0]
        del row_of
        # annulus element blocks: positions of their local (24 x 24) entries in the CSR data
        nd = fe.num_nodes * fe.vec
        sel = np.asarray(pp.sel)
        local = (sel[:, None] * nd * nd + np.arange(nd * nd)[None, :]).ravel()
        self.annulus_pos = jnp.asarray(self.coo_to_csr[local])
        self.nn, self.vec = fe.num_nodes, fe.vec
        self.sd = sd
        self.omega0 = problem._omega
        self.pp = pp

    def rest_data(self):
        """CSR data of K_rest at the current coefficients of ``pp.base`` (annulus material set to zero)."""
        pp = self.pp
        nsel, nq = len(pp.sel), pp.nq
        pp.set_quadrature_material(jnp.zeros((nsel, nq, 2, 2, 2, 2)), jnp.zeros((nsel, nq)))
        sol0 = [np.zeros((f.num_total_nodes, f.vec)) for f in pp.problem.fes]
        pp.problem.newton_update(sol0)
        data = np.bincount(self.coo_to_csr, weights=np.asarray(pp.problem.V), minlength=self.nnz)
        data[self.bc_entries] = 0.0
        data[self.bc_diag] = 1.0
        return jnp.asarray(data)

    def annulus_blocks(self, C_q, rho_q):
        """Element matrices of the annulus, (n_sel, 24, 24), in jax-fem DOF order [node][Re x, Re y, Im x, Im y]."""
        g, w, N = self.sd.grads, self.sd.jxw, self.sd.vals
        K = jnp.einsum("cq,cqaj,cqijkl,cqbl->caibk", w, g, C_q, g)
        M = jnp.einsum("cq,cq,qa,qb->cab", w, rho_q, N, N)
        K = K - self.omega0 ** 2 * M[:, :, None, :, None] * jnp.eye(2)[None, None, :, None, :]
        nc, nn = K.shape[0], self.nn
        E = jnp.zeros((nc, nn, 4, nn, 4))
        E = E.at[:, :, :2, :, :2].set(K).at[:, :, 2:, :, 2:].set(K)    # no damping in the annulus
        return E.reshape(nc, nn * 4, nn * 4)

    def data(self, rest, C_q, rho_q):
        blocks = self.annulus_blocks(C_q, rho_q).reshape(-1)
        return rest.at[self.annulus_pos].add(blocks)

    def to_scipy(self, data):
        import scipy.sparse as sp
        return sp.csr_matrix((np.asarray(data), self.indices, self.indptr), shape=(self.n, self.n))
