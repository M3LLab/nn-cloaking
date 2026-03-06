"""Test Voigt conversion roundtrip: C → Voigt4 → C and M → C → Voigt4.

The augmented Voigt notation uses 4 pairs: (0,0), (1,1), (0,1), (1,0),
mapping a (2,2,2,2) tensor to a (4,4) matrix and back.

Key concern: because pairs (0,1) and (1,0) are SEPARATE entries (indices 2 and 3),
the mapping must be bijective for tensors without minor symmetry.
If the tensor HAS minor symmetry (C[i,j,k,l] = C[j,i,k,l]), rows/cols 2 and 3
of the Voigt matrix will be identical — this is fine but worth checking.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

from boundaries import (
    C_to_voigt4, voigt4_to_C, C0, C_eff, symmetrize_stiffness,
    _in_cloak, x_c, y_top, a, b, c,
)

TOL = 1e-12


def get_cloak_point():
    """Return a point inside the cloak region."""
    _a, _b = float(a), float(b)
    depth = (_a + _b) / 2.0
    y = y_top - depth
    pt = jnp.array([x_c + 0.1 * float(c), y])
    assert _in_cloak(pt), f"Test point {pt} not in cloak!"
    return pt


def test_roundtrip_C0():
    """C0 → Voigt → C should recover C0."""
    print("\n  Test: C0 roundtrip (C → Voigt4 → C)")
    M = C_to_voigt4(C0)
    C_recovered = voigt4_to_C(M)
    err = float(jnp.max(jnp.abs(C_recovered - C0)))
    passed = err < TOL
    print(f"    max|C_recovered - C0| = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"    C0:\n{C0}")
        print(f"    Recovered:\n{C_recovered}")
    return passed


def test_roundtrip_ceff_raw():
    """Raw C_eff (no symmetrization) roundtrip."""
    print("\n  Test: raw C_eff roundtrip (C → Voigt4 → C)")
    pt = get_cloak_point()
    Ce = C_eff(pt, symmetrize=False)
    M = C_to_voigt4(Ce)
    C_recovered = voigt4_to_C(M)
    err = float(jnp.max(jnp.abs(C_recovered - Ce)))
    passed = err < TOL
    print(f"    max|C_recovered - C_eff| = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_roundtrip_ceff_symmetrized():
    """Symmetrized C_eff roundtrip."""
    print("\n  Test: symmetrized C_eff roundtrip (C → Voigt4 → C)")
    pt = get_cloak_point()
    Ce = C_eff(pt, symmetrize=True)
    M = C_to_voigt4(Ce)
    C_recovered = voigt4_to_C(M)
    err = float(jnp.max(jnp.abs(C_recovered - Ce)))
    passed = err < TOL
    print(f"    max|C_recovered - C_eff| = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_roundtrip_voigt_matrix():
    """Random 4×4 matrix M → C → Voigt4 should recover M."""
    print("\n  Test: random matrix roundtrip (Voigt4 → C → Voigt4)")
    key = jax.random.PRNGKey(42)
    M = jax.random.normal(key, (4, 4))
    C = voigt4_to_C(M)
    M_recovered = C_to_voigt4(C)
    err = float(jnp.max(jnp.abs(M_recovered - M)))
    passed = err < TOL
    print(f"    max|M_recovered - M| = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"    M:\n{M}")
        print(f"    Recovered:\n{M_recovered}")
    return passed


def test_voigt_structure_C0():
    """Check that the Voigt matrix of C0 has expected isotropic structure."""
    print("\n  Test: Voigt matrix structure of C0")
    M = C_to_voigt4(C0)
    print(f"    Voigt4(C0) =")
    for row in range(4):
        print(f"      [{', '.join(f'{float(M[row,col]):12.2f}' for col in range(4))}]")

    # For isotropic C0 with Voigt pairs (00,11,01,10):
    # M should be symmetric (since C0 has full symmetry)
    sym_err = float(jnp.max(jnp.abs(M - M.T)))
    print(f"    max|M - M.T| = {sym_err:.2e}  (should be 0 for isotropic)")

    # Rows/cols 2 and 3 should be identical (minor symmetry of C0)
    row_diff = float(jnp.max(jnp.abs(M[2, :] - M[3, :])))
    col_diff = float(jnp.max(jnp.abs(M[:, 2] - M[:, 3])))
    print(f"    max|row2 - row3| = {row_diff:.2e}  (should be 0, minor sym)")
    print(f"    max|col2 - col3| = {col_diff:.2e}  (should be 0, minor sym)")

    passed = sym_err < TOL and row_diff < TOL and col_diff < TOL
    print(f"    {'PASS' if passed else 'FAIL'}")
    return passed


def test_voigt_structure_ceff_raw():
    """Check Voigt matrix of raw C_eff — expect broken minor symmetry."""
    print("\n  Test: Voigt matrix structure of raw C_eff")
    pt = get_cloak_point()
    Ce = C_eff(pt, symmetrize=False)
    M = C_to_voigt4(Ce)
    print(f"    Voigt4(C_eff_raw) =")
    for row in range(4):
        print(f"      [{', '.join(f'{float(M[row,col]):12.2f}' for col in range(4))}]")

    sym_err = float(jnp.max(jnp.abs(M - M.T)))
    row_diff = float(jnp.max(jnp.abs(M[2, :] - M[3, :])))
    col_diff = float(jnp.max(jnp.abs(M[:, 2] - M[:, 3])))
    print(f"    max|M - M.T| = {sym_err:.2e}")
    print(f"    max|row2 - row3| = {row_diff:.2e}  (>0 = broken minor sym)")
    print(f"    max|col2 - col3| = {col_diff:.2e}  (>0 = broken minor sym)")

    # We EXPECT this to fail minor symmetry for raw C_eff
    if row_diff > TOL or col_diff > TOL:
        print(f"    INFO: Minor symmetry IS broken (as expected for raw Cosserat)")
    return True  # This is informational, not a pass/fail


if __name__ == "__main__":
    print("=" * 70)
    print("  DIAGNOSTIC: Voigt conversion roundtrip tests")
    print("=" * 70)

    results = {}
    results["C0_roundtrip"] = test_roundtrip_C0()
    results["ceff_raw_roundtrip"] = test_roundtrip_ceff_raw()
    results["ceff_sym_roundtrip"] = test_roundtrip_ceff_symmetrized()
    results["voigt_matrix_roundtrip"] = test_roundtrip_voigt_matrix()
    results["voigt_structure_C0"] = test_voigt_structure_C0()
    test_voigt_structure_ceff_raw()  # informational only

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
