"""Test whether the stress tensor σ_ij = C_ijkl ε_kl is symmetric.

For standard (Cauchy) elasticity, σ must be symmetric: σ_ij = σ_ji.
This requires the minor symmetry C_ijkl = C_jikl.

In Cosserat elasticity, asymmetric stress is allowed, but standard FEM
with symmetric test/trial functions implicitly assumes symmetric stress.
If C_eff lacks minor symmetry, the FEM assembles an inconsistent system.

Tests:
1. Synthetic strains → stress symmetry check (raw vs symmetrized C_eff)
2. Quantify the asymmetric part of stress: σ_anti = (σ - σ^T) / 2
3. Check how asymmetry varies spatially across the cloak
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from boundaries import (
    C_eff, C0, _in_cloak, F_tensor,
    x_c, y_top, a, b, c, mesh,
)

TOL = 1e-10


def get_cloak_points(n=5):
    """Generate points inside the cloak at various positions."""
    _a, _b, _c = float(a), float(b), float(c)
    pts = []
    for frac_depth in np.linspace(0.2, 0.8, n):
        depth = _a + frac_depth * (_b - _a)
        y = y_top - depth
        r_max = 1.0 - depth / _b
        for frac_r in [0.2, 0.5]:
            for side in [+1, -1]:
                dx = side * frac_r * r_max * _c
                pt = jnp.array([x_c + dx, y])
                if _in_cloak(pt):
                    pts.append(pt)
    return pts


def compute_stress(C, strain):
    """σ_ij = C_ijkl ε_kl"""
    return jnp.einsum('ijkl,kl->ij', C, strain)


def stress_asymmetry(sigma):
    """Returns the antisymmetric part norm relative to full stress norm."""
    anti = 0.5 * (sigma - sigma.T)
    sym = 0.5 * (sigma + sigma.T)
    norm_anti = jnp.linalg.norm(anti)
    norm_full = jnp.linalg.norm(sigma)
    return float(norm_anti), float(norm_full)


def test_synthetic_strains(symmetrize: bool):
    """Apply several canonical strain states and check stress symmetry."""
    label = "symmetrized" if symmetrize else "raw"
    print(f"\n{'='*70}")
    print(f"  STRESS SYMMETRY — synthetic strains, C_eff {label}")
    print(f"{'='*70}")

    # Canonical strain states
    strains = {
        "uniaxial_xx": jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        "uniaxial_yy": jnp.array([[0.0, 0.0], [0.0, 1.0]]),
        "pure_shear":  jnp.array([[0.0, 0.5], [0.5, 0.0]]),
        "simple_shear": jnp.array([[0.0, 1.0], [0.0, 0.0]]),  # asymmetric gradient
    }

    pts = get_cloak_points(n=3)
    if not pts:
        print("  ERROR: No cloak points found!")
        return False

    all_pass = True
    for strain_name, eps in strains.items():
        max_rel_asym = 0.0
        worst_pt = None
        worst_sigma = None

        for pt in pts:
            Ce = C_eff(pt, symmetrize=symmetrize)
            sigma = compute_stress(Ce, eps)
            norm_anti, norm_full = stress_asymmetry(sigma)
            rel_asym = norm_anti / max(norm_full, 1e-30)
            if rel_asym > max_rel_asym:
                max_rel_asym = rel_asym
                worst_pt = pt
                worst_sigma = sigma

        passed = max_rel_asym < TOL
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] strain={strain_name}, max relative asymmetry = {max_rel_asym:.6e}")
        if not passed and worst_sigma is not None:
            print(f"         at pt=({float(worst_pt[0]):.4f}, {float(worst_pt[1]):.4f})")
            print(f"         σ = [[{float(worst_sigma[0,0]):.4e}, {float(worst_sigma[0,1]):.4e}],")
            print(f"              [{float(worst_sigma[1,0]):.4e}, {float(worst_sigma[1,1]):.4e}]]")
            print(f"         σ_12 - σ_21 = {float(worst_sigma[0,1] - worst_sigma[1,0]):.4e}")

    return all_pass


def test_spatial_asymmetry_map():
    """Map stress asymmetry across the cloak to see spatial pattern."""
    print(f"\n{'='*70}")
    print(f"  STRESS ASYMMETRY SPATIAL MAP (pure shear strain, raw C_eff)")
    print(f"{'='*70}")

    eps = jnp.array([[0.0, 0.5], [0.5, 0.0]])  # symmetric pure shear
    _a, _b, _c = float(a), float(b), float(c)

    print(f"\n  {'depth':>8s}  {'dx':>8s}  {'|σ_anti|/|σ|':>14s}  {'σ12':>12s}  {'σ21':>12s}  {'σ12-σ21':>12s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*12}")

    for frac_depth in np.linspace(0.1, 0.9, 5):
        depth = _a + frac_depth * (_b - _a)
        y = y_top - depth
        r_max = 1.0 - depth / _b
        for frac_r in [0.1, 0.3, 0.5, 0.7]:
            dx = frac_r * r_max * _c
            pt = jnp.array([x_c + dx, y])
            if not _in_cloak(pt):
                continue
            Ce = C_eff(pt, symmetrize=False)
            sigma = compute_stress(Ce, eps)
            norm_anti, norm_full = stress_asymmetry(sigma)
            rel = norm_anti / max(norm_full, 1e-30)
            print(f"  {depth:8.4f}  {dx:8.4f}  {rel:14.6e}  "
                  f"{float(sigma[0,1]):12.4e}  {float(sigma[1,0]):12.4e}  "
                  f"{float(sigma[0,1]-sigma[1,0]):12.4e}")


def test_outside_cloak():
    """Verify stress is symmetric outside the cloak (C_eff = C0 there)."""
    print(f"\n{'='*70}")
    print(f"  STRESS SYMMETRY OUTSIDE CLOAK (should always pass)")
    print(f"{'='*70}")

    eps = jnp.array([[0.0, 0.5], [0.5, 0.0]])
    # Point far from cloak
    pt = jnp.array([float(x_c) + 3.0 * float(c), float(y_top) - 0.5])
    Ce = C_eff(pt, symmetrize=False)
    sigma = compute_stress(Ce, eps)
    norm_anti, norm_full = stress_asymmetry(sigma)
    rel = norm_anti / max(norm_full, 1e-30)
    passed = rel < TOL
    print(f"  [{'PASS' if passed else 'FAIL'}] outside cloak: relative asymmetry = {rel:.2e}")
    return passed


if __name__ == "__main__":
    print("=" * 70)
    print("  DIAGNOSTIC: Stress symmetry tests")
    print("=" * 70)

    results = {}
    results["stress_sym_raw"] = test_synthetic_strains(symmetrize=False)
    results["stress_sym_symmetrized"] = test_synthetic_strains(symmetrize=True)
    results["outside_cloak"] = test_outside_cloak()
    test_spatial_asymmetry_map()

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")

    if all(results.values()):
        print("\n  All tests passed.")
    else:
        print("\n  Some tests FAILED — see details above.")
        sys.exit(1)
