"""Compare numerical C_eff (einsum-based) with analytical closed-form formula.

This test is self-contained: it copies the core functions from boundaries.py
to avoid triggering the full mesh generation + solve at import time.
"""

import jax.numpy as jnp
import numpy as np

# ── Parameters from boundaries.py ──
rho0 = 1600.0
cs   = 300.0
cp   = np.sqrt(3.0) * cs
mu   = rho0 * cs**2 * 0.1
lam  = rho0 * cp**2 - 2 * mu
nu   = lam / (2 * (lam + mu))
cR   = cs * (0.826 + 1.14 * nu) / (1 + nu)
f_star      = 2.0
lambda_star = 1.0
H = 4.305 * lambda_star
W = 12.5  * lambda_star
a   = 0.0774 * H
b   = 3 * a
c   = 0.309 * H / 2.0
L_pml = 1.0 * lambda_star
W_total = 2 * L_pml + W
H_total = L_pml + H
x_off = L_pml
y_off = L_pml
y_top = H_total
x_c = x_off + W / 2.0

# ── Core functions from boundaries.py ──

def _in_cloak(x):
    depth = y_top - x[1]
    r     = jnp.abs(x[0] - x_c) / c
    d2    = b * (1.0 - r)
    d1    = a * (1.0 - r)
    return (r <= 1.0) & (depth >= d1) & (depth <= d2)

def F_tensor(x):
    sign = jnp.where(x[0] >= x_c, 1.0, -1.0)
    F21  = sign * a / c
    F22  = (b - a) / b
    F_cloak = jnp.array([[1.0, 0.0],
                          [F21, F22]])
    return jnp.where(_in_cloak(x), F_cloak, jnp.eye(2))

def C_iso(lam_, mu_):
    I = jnp.eye(2)
    term_lam = lam_ * jnp.einsum("ij,kl->ijkl", I, I)
    term_mu  = mu_ * (jnp.einsum("ik,jl->ijkl", I, I) + jnp.einsum("il,jk->ijkl", I, I))
    return term_lam + term_mu

C0 = C_iso(lam, mu)

pairs = [(0,0), (1,1), (0,1), (1,0)]

def C_to_voigt4(C):
    M = jnp.zeros((4,4))
    for I_,(i,j) in enumerate(pairs):
        for J_,(k,l) in enumerate(pairs):
            M = M.at[I_,J_].set(C[i,j,k,l])
    return M

def C_eff_numerical(x):
    """Einsum-based C_eff from boundaries.py."""
    F   = F_tensor(x)
    J   = jnp.linalg.det(F)
    Cnew = jnp.einsum('iI,kK,IjKl->ijkl', F, F, C0) / J
    return jnp.where(_in_cloak(x), Cnew, C0)

def C_eff_analytical(x):
    """Closed-form 4x4 augmented-Voigt from debug_ceff_elements.py."""
    F = F_tensor(x)
    F21 = F[1, 0]
    F22 = F[1, 1]

    return jnp.array([
        [(lam + 2*mu)/F22,
         lam,
         0.0,
         (F21/F22)*(lam + 2*mu)],

        [lam,
         (F21**2 * mu + F22**2 * (lam + 2*mu)) / F22,
         (F21/F22)*mu,
         F21*(lam + mu)],

        [0.0,
         (F21/F22)*mu,
         mu/F22,
         mu],

        [(F21/F22)*(lam + 2*mu),
         F21*(lam + mu),
         mu,
         (F21**2 * (lam + 2*mu) + F22**2 * mu) / F22]
    ])

# ── Helper to get a point inside the cloak ──

def _cloak_point(r_test=0.3, side="right"):
    """Return a point inside the cloak at normalised lateral distance r_test."""
    sign = 1.0 if side == "right" else -1.0
    x_test = x_c + sign * r_test * c
    depth_mid = 0.5 * (a * (1 - r_test) + b * (1 - r_test))
    y_test = y_top - depth_mid
    return jnp.array([x_test, y_test])


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════

def test_ceff_matches_analytical():
    """C_eff via einsum must match the closed-form 4x4 Voigt matrix."""
    pt = _cloak_point(r_test=0.3, side="right")

    C_num_voigt = C_to_voigt4(C_eff_numerical(pt))
    C_ana_voigt = C_eff_analytical(pt)

    print("Numerical (4x4 Voigt):")
    print(np.array(C_num_voigt))
    print("\nAnalytical (4x4 Voigt):")
    print(np.array(C_ana_voigt))
    print(f"\nMax abs error: {float(jnp.max(jnp.abs(C_num_voigt - C_ana_voigt))):.2e}")
    print(f"Max rel error: {float(jnp.max(jnp.abs(C_num_voigt - C_ana_voigt) / (jnp.abs(C_ana_voigt) + 1e-30))):.2e}")

    assert jnp.allclose(C_num_voigt, C_ana_voigt, atol=1e-6, rtol=1e-6), \
        "C_eff numerical does NOT match analytical!"
    print("✓ PASS")


def test_ceff_multiple_r_values():
    """Test at several r values across the cloak."""
    for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for side in ["left", "right"]:
            pt = _cloak_point(r_test=r, side=side)
            C_num = C_to_voigt4(C_eff_numerical(pt))
            C_ana = C_eff_analytical(pt)
            err = float(jnp.max(jnp.abs(C_num - C_ana)))
            print(f"  r={r:.1f} {side:5s}  max_err={err:.2e}")
            assert jnp.allclose(C_num, C_ana, atol=1e-6, rtol=1e-6), \
                f"Mismatch at r={r}, side={side}"
    print("✓ PASS")


def test_ceff_outside_cloak_is_C0():
    """Outside the cloak, C_eff should return the isotropic C0."""
    pt = jnp.array([1.0, 1.0])  # far from cloak
    C_out = C_eff_numerical(pt)
    assert jnp.allclose(C_out, C0, atol=1e-10), "C_eff outside cloak should be C0"
    print("✓ PASS")


def test_ceff_diagonal_symmetry():
    """Diagonal entries of C_eff should be the same on left vs right side."""
    pt_right = _cloak_point(0.3, "right")
    pt_left  = _cloak_point(0.3, "left")

    C_right = C_to_voigt4(C_eff_numerical(pt_right))
    C_left  = C_to_voigt4(C_eff_numerical(pt_left))

    print(f"  Right diag: {np.diag(np.array(C_right))}")
    print(f"  Left  diag: {np.diag(np.array(C_left))}")
    assert jnp.allclose(jnp.diag(C_right), jnp.diag(C_left), atol=1e-10), \
        "Diagonal of C_eff should match on both sides"
    print("✓ PASS")


if __name__ == "__main__":
    print("=== test_ceff_matches_analytical ===")
    test_ceff_matches_analytical()
    print("\n=== test_ceff_multiple_r_values ===")
    test_ceff_multiple_r_values()
    print("\n=== test_ceff_outside_cloak_is_C0 ===")
    test_ceff_outside_cloak_is_C0()
    print("\n=== test_ceff_diagonal_symmetry ===")
    test_ceff_diagonal_symmetry()
    print("\nAll tests passed.")
