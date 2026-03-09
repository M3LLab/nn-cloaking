"""Material tensor computations for transformational elasticity.

All functions are pure (no global state) and JAX-traceable where needed.
"""

from __future__ import annotations

import jax.numpy as jnp

from rayleigh_cloak.geometry.base import CloakGeometry


# ── isotropic stiffness ──────────────────────────────────────────────

def C_iso(lam: float, mu: float) -> jnp.ndarray:
    """Isotropic 2-D stiffness tensor C_{ijkl} (plane-strain)."""
    I = jnp.eye(2)
    return (
        lam * jnp.einsum("ij,kl->ijkl", I, I)
        + mu * (jnp.einsum("ik,jl->ijkl", I, I)
                + jnp.einsum("il,jk->ijkl", I, I))
    )

# ── Voigt (3×3) conversions ────────────────────────────────
def C_to_voigt3(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to 3×3 Voigt matrix."""
    M = jnp.zeros((3, 3))
    M = M.at[0, 0].set(C[0, 0, 0, 0])
    M = M.at[1, 1].set(C[1, 1, 1, 1])
    M = M.at[2, 2].set(C[0, 1, 0, 1])
    M = M.at[0, 1].set(C[0, 0, 1, 1])
    M = M.at[1, 0].set(C[1, 1, 0, 0])
    return M

def voigt3_to_C(M: jnp.ndarray) -> jnp.ndarray:
    """Convert 3×3 Voigt matrix to (2,2,2,2) tensor.

    Sets all four shear components to M[2,2] (minor symmetry).
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(M[0, 0])
    C = C.at[1, 1, 1, 1].set(M[1, 1])
    C = C.at[0, 1, 0, 1].set(M[2, 2])
    C = C.at[1, 0, 1, 0].set(M[2, 2])
    C = C.at[0, 1, 1, 0].set(M[2, 2])
    C = C.at[1, 0, 0, 1].set(M[2, 2])
    C = C.at[0, 0, 1, 1].set(M[0, 1])
    C = C.at[1, 1, 0, 0].set(M[1, 0])
    return C

def voight3_to_flatC(M: jnp.ndarray) -> jnp.ndarray:
    """Convert 3×3 Voigt matrix to flattened tensor (3,3) → (4,)."""
    return jnp.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1]])

def flatC_to_voigt3(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert flattened tensor (4,) to 3×3 Voigt matrix."""
    M = jnp.zeros((3, 3))
    M = M.at[0, 0].set(flat[0])
    M = M.at[1, 1].set(flat[1])
    M = M.at[2, 2].set(flat[2])
    M = M.at[0, 1].set(flat[3])
    M = M.at[1, 0].set(flat[3])
    return M

def C_to_flatC(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to flattened (4,) array.

    .. warning:: The 4-param representation enforces minor symmetry, producing
       a rank-3 augmented Voigt matrix.  This is **incompatible** with the
       full-gradient (Cosserat) FEM formulation.  Use ``n_C_params=6`` or
       ``n_C_params=16`` instead.
    """
    flat = jnp.zeros(4)
    flat = flat.at[0].set(C[0, 0, 0, 0])
    flat = flat.at[1].set(C[1, 1, 1, 1])
    flat = flat.at[2].set(C[0, 1, 0, 1])
    flat = flat.at[3].set(C[0, 0, 1, 1])
    return flat

def flatC_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert flattened (4,) array to (2,2,2,2) tensor.

    Enforces minor symmetry on the shear block: all four components
    C[0,1,0,1], C[1,0,1,0], C[0,1,1,0], C[1,0,0,1] are set to flat[2].

    .. warning:: Produces a rank-3 augmented Voigt matrix — singular for the
       full-gradient FEM formulation.
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(flat[0])
    C = C.at[1, 1, 1, 1].set(flat[1])
    C = C.at[0, 1, 0, 1].set(flat[2])
    C = C.at[1, 0, 1, 0].set(flat[2])
    C = C.at[0, 1, 1, 0].set(flat[2])
    C = C.at[1, 0, 0, 1].set(flat[2])
    C = C.at[0, 0, 1, 1].set(flat[3])
    C = C.at[1, 1, 0, 0].set(flat[3])
    return C


# ── 6-param (block-diagonal Cosserat) ────────────────────────────────

def C_to_flat6(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to 6-param flat array.

    Captures the block-diagonal structure of the augmented 4×4 Voigt matrix,
    preserving the Cosserat asymmetry in the shear block::

        flat[0] = C[0,0,0,0]   (normal-xx)
        flat[1] = C[1,1,1,1]   (normal-yy)
        flat[2] = C[0,0,1,1]   (normal coupling, = C[1,1,0,0] by major sym.)
        flat[3] = C[0,1,0,1]   (shear diagonal)
        flat[4] = C[1,0,1,0]   (shear diagonal)
        flat[5] = C[0,1,1,0]   (shear off-diag, = C[1,0,0,1] by major sym.)
    """
    return jnp.array([
        C[0, 0, 0, 0],
        C[1, 1, 1, 1],
        C[0, 0, 1, 1],
        C[0, 1, 0, 1],
        C[1, 0, 1, 0],
        C[0, 1, 1, 0],
    ])


def flat6_to_C(flat: jnp.ndarray) -> jnp.ndarray:
    """Convert 6-param flat array to (2,2,2,2) tensor.

    Produces the block-diagonal augmented Voigt::

        [[flat[0]  flat[2]  0        0       ]
         [flat[2]  flat[1]  0        0       ]
         [0        0        flat[3]  flat[5] ]
         [0        0        flat[5]  flat[4] ]]

    Major symmetry is preserved; the shear block has full rank when
    ``flat[3]*flat[4] != flat[5]**2``.
    """
    C = jnp.zeros((2, 2, 2, 2))
    C = C.at[0, 0, 0, 0].set(flat[0])
    C = C.at[1, 1, 1, 1].set(flat[1])
    C = C.at[0, 0, 1, 1].set(flat[2])
    C = C.at[1, 1, 0, 0].set(flat[2])
    C = C.at[0, 1, 0, 1].set(flat[3])
    C = C.at[1, 0, 1, 0].set(flat[4])
    C = C.at[0, 1, 1, 0].set(flat[5])
    C = C.at[1, 0, 0, 1].set(flat[5])
    return C

# ── augmented Voigt (4×4) conversions ────────────────────────────────

# Augmented Voigt index map: 0→(0,0), 1→(1,1), 2→(0,1), 3→(1,0)
_PAIRS = [(0, 0), (1, 1), (0, 1), (1, 0)]


def C_to_voigt4(C: jnp.ndarray) -> jnp.ndarray:
    """Convert (2,2,2,2) tensor to augmented 4×4 Voigt matrix."""
    M = jnp.zeros((4, 4))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            M = M.at[I, J].set(C[i, j, k, l])
    return M


def voigt4_to_C(M: jnp.ndarray) -> jnp.ndarray:
    """Convert augmented 4×4 Voigt matrix to (2,2,2,2) tensor."""
    C = jnp.zeros((2, 2, 2, 2))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            C = C.at[i, j, k, l].set(M[I, J])
    return C


# ── symmetrisation (Chatzopoulos et al.) ────────────────────────────


def symmetrize_stiffness(Ceff: jnp.ndarray) -> jnp.ndarray:
    """Symmetrise an effective stiffness using the augmented-Voigt recipe.

    Csym_IJ = (C_IJ + C_{Ī J} + C_{I J̄} + C_{Ī J̄}) / 4

    where bar swaps indices 2↔3 (i.e. (0,1)↔(1,0)).
    """
    M = C_to_voigt4(Ceff)
    P = jnp.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]])

    Msym = (M + P @ M + M @ P + P @ M @ P) / 4.0
    Msym = 0.5 * (Msym + Msym.T)  # enforce Cauchy symmetry
    return voigt4_to_C(Msym)


# ── position-dependent effective properties ──────────────────────────


def C_eff(
    x: jnp.ndarray,
    geometry: CloakGeometry,
    C0: jnp.ndarray,
    symmetrize: bool = False,
) -> jnp.ndarray:
    """Effective stiffness tensor at position *x*."""
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    Cnew = jnp.einsum("jJ,lL,iJkL->ijkl", F, F, C0) / J
    if symmetrize:
        Cnew = symmetrize_stiffness(Cnew)
    return jnp.where(geometry.in_cloak(x), Cnew, C0)


def rho_eff(
    x: jnp.ndarray,
    geometry: CloakGeometry,
    rho0: float,
) -> jnp.ndarray:
    """Effective density at position *x*."""
    F = geometry.F_tensor(x)
    J = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    return jnp.where(geometry.in_cloak(x), rho_cloak, rho0)


from rayleigh_cloak.cells import CellDecomposition


# ── flat↔tensor converters keyed by n_C_params ───────────────────────

def _get_converters(n_C_params: int):
    """Return ``(to_flat, from_flat)`` functions for the chosen parameterisation.

    Supported values:

    * **6** — block-diagonal Cosserat (recommended minimum).  Captures the
      normal block and the full 2×2 shear block of the augmented Voigt matrix.
    * **16** — full augmented 4×4 Voigt (most general).
    * **4** — minor-symmetric shear (deprecated, produces rank-3 Voigt →
      singular stiffness in the full-gradient FEM formulation).
    """
    if n_C_params == 4:
        import warnings
        warnings.warn(
            "n_C_params=4 enforces minor symmetry, producing a rank-3 "
            "augmented Voigt matrix.  This leads to a singular stiffness "
            "matrix in the Cosserat FEM formulation.  Use n_C_params=6 or 16.",
            stacklevel=2,
        )
        return C_to_flatC, flatC_to_C
    if n_C_params == 6:
        return C_to_flat6, flat6_to_C
    if n_C_params == 16:
        def _to16(C):
            return C_to_voigt4(C).ravel()
        def _from16(flat):
            return voigt4_to_C(flat.reshape(4, 4))
        return _to16, _from16
    raise ValueError(f"Unsupported n_C_params={n_C_params} (use 6 or 16)")


# ── cell-based material model ────────────────────────────────────────


class CellMaterial:
    """Piecewise-constant material over a :class:`CellDecomposition`.

    Parameters
    ----------
    geometry : CloakGeometry
    C0 : (2,2,2,2) background stiffness
    rho0 : background density
    cell_decomp : CellDecomposition
    n_C_params : number of flat parameters per cell (6 or 16)
    """

    def __init__(
        self,
        geometry: CloakGeometry,
        C0: jnp.ndarray,
        rho0: float,
        cell_decomp: CellDecomposition,
        n_C_params: int = 4,
    ):
        self.geometry = geometry
        self.C0 = C0
        self.rho0 = rho0
        self.cell_decomp = cell_decomp
        self.n_C_params = n_C_params
        self.to_flat, self.from_flat = _get_converters(n_C_params)

        self.cell_C_flat, self.cell_rho = self._initialize()

    def _initialize(self):
        """Compute initial per-cell C and rho from the continuous push-forward."""
        centers = self.cell_decomp.cell_centers  # (n_cells, 2)
        mask = self.cell_decomp.cloak_mask        # (n_cells,)

        C0_flat = self.to_flat(self.C0)

        cell_C_list = []
        cell_rho_list = []
        for i, center in enumerate(centers):
            if mask[i]:
                x = jnp.array(center)
                C_i = C_eff(x, self.geometry, self.C0)
                C_i = symmetrize_stiffness(C_i)  # enforce minor symmetry for stability
                cell_C_list.append(self.to_flat(C_i))
                cell_rho_list.append(float(rho_eff(x, self.geometry, self.rho0)))
            else:
                cell_C_list.append(C0_flat)
                cell_rho_list.append(self.rho0)

        return jnp.stack(cell_C_list), jnp.array(cell_rho_list)

    def get_initial_params(self):
        """Return ``(cell_C_flat, cell_rho)`` as a JAX pytree."""
        return (self.cell_C_flat, self.cell_rho)
