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
