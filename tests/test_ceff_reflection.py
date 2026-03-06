"""Test C_eff left-right reflection symmetry about the cloak axis x = x_c.

Under reflection R = diag(-1, 1) about x_c, a 4th-order tensor transforms as:
    C'[i,j,k,l] = R[i,i'] R[j,j'] R[k,k'] R[l,l'] C[i',j',k',l']

Since R = diag(-1,1), each index 0 (x-direction) contributes a factor of -1.
So components with an EVEN number of 0-indices are invariant,
and components with an ODD number of 0-indices flip sign.

For the cloak to behave symmetrically, C_eff at mirror points must satisfy
this relation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import numpy as np

# boundaries.py runs mesh generation at import time — that's fine for tests
from boundaries import (
    C_eff, C0, _in_cloak, F_tensor, symmetrize_stiffness,
    x_c, y_top, a, b, c,
)

TOL = 1e-10


def count_x_indices(i, j, k, l):
    """Count how many indices are 0 (x-direction)."""
    return sum(1 for idx in (i, j, k, l) if idx == 0)


def reflection_sign(i, j, k, l):
    """Expected sign factor under x-reflection: (-1)^(count of x-indices)."""
    return (-1) ** count_x_indices(i, j, k, l)


def get_test_mirror_points():
    """Generate pairs of mirror points (x_c+dx, y) and (x_c-dx, y) inside the cloak."""
    _a, _b, _c = float(a), float(b), float(c)
    pairs = []

    # Several points at different depths and lateral offsets
    for frac_depth in [0.3, 0.5, 0.7]:  # fraction between inner (a) and outer (b) depth
        depth = _a + frac_depth * (_b - _a)
        y = y_top - depth
        # At this depth, the cloak extends laterally to r_max
        r_max = 1.0 - depth / _b  # from outer boundary: depth = b*(1-r) → r = 1 - depth/b
        for frac_r in [0.1, 0.3, 0.5]:
            dx = frac_r * r_max * _c
            pt_R = jnp.array([x_c + dx, y])
            pt_L = jnp.array([x_c - dx, y])
            # Verify both points are actually in cloak
            if _in_cloak(pt_R) and _in_cloak(pt_L):
                pairs.append((pt_R, pt_L, f"depth={depth:.4f}, dx={dx:.4f}"))
    return pairs


def test_reflection_symmetry(symmetrize: bool):
    """Test that C_eff at mirror points satisfies reflection symmetry."""
    label = "symmetrized" if symmetrize else "raw (unsymmetrized)"
    print(f"\n{'='*70}")
    print(f"  C_eff REFLECTION SYMMETRY — {label}")
    print(f"{'='*70}")

    pairs = get_test_mirror_points()
    if not pairs:
        print("  ERROR: No valid mirror point pairs found inside cloak!")
        return False

    all_pass = True
    for pt_R, pt_L, desc in pairs:
        C_R = C_eff(pt_R, symmetrize=symmetrize)
        C_L = C_eff(pt_L, symmetrize=symmetrize)

        point_pass = True
        failures = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        sign = reflection_sign(i, j, k, l)
                        expected = sign * float(C_R[i, j, k, l])
                        actual = float(C_L[i, j, k, l])
                        if abs(expected - actual) > TOL:
                            point_pass = False
                            failures.append(
                                f"    C[{i},{j},{k},{l}]: "
                                f"C_R={float(C_R[i,j,k,l]):.8e}, "
                                f"C_L={actual:.8e}, "
                                f"expected C_L={expected:.8e} "
                                f"(sign={sign:+d}), "
                                f"err={abs(expected-actual):.2e}"
                            )

        status = "PASS" if point_pass else "FAIL"
        print(f"\n  [{status}] {desc}")
        print(f"         pt_R=({float(pt_R[0]):.4f}, {float(pt_R[1]):.4f}), "
              f"pt_L=({float(pt_L[0]):.4f}, {float(pt_L[1]):.4f})")
        if not point_pass:
            all_pass = False
            for f in failures:
                print(f)

    return all_pass


def test_minor_symmetries(symmetrize: bool):
    """Test C_eff[i,j,k,l] == C_eff[j,i,k,l] and C_eff[i,j,k,l] == C_eff[i,j,l,k]."""
    label = "symmetrized" if symmetrize else "raw (unsymmetrized)"
    print(f"\n{'='*70}")
    print(f"  C_eff MINOR SYMMETRIES — {label}")
    print(f"{'='*70}")

    pairs = get_test_mirror_points()
    # Just use the right-side points
    test_pts = [pt_R for pt_R, _, _ in pairs]
    if not test_pts:
        print("  ERROR: No test points found!")
        return False

    all_pass = True
    for pt in test_pts:
        Ce = C_eff(pt, symmetrize=symmetrize)
        failures = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        # Minor symmetry 1: swap i <-> j
                        if abs(float(Ce[i,j,k,l] - Ce[j,i,k,l])) > TOL:
                            failures.append(
                                f"    C[{i},{j},{k},{l}]={float(Ce[i,j,k,l]):.8e} != "
                                f"C[{j},{i},{k},{l}]={float(Ce[j,i,k,l]):.8e} (minor sym 1)"
                            )
                        # Minor symmetry 2: swap k <-> l
                        if abs(float(Ce[i,j,k,l] - Ce[i,j,l,k])) > TOL:
                            failures.append(
                                f"    C[{i},{j},{k},{l}]={float(Ce[i,j,k,l]):.8e} != "
                                f"C[{i},{j},{l},{k}]={float(Ce[i,j,l,k]):.8e} (minor sym 2)"
                            )

        point_pass = len(failures) == 0
        status = "PASS" if point_pass else "FAIL"
        print(f"  [{status}] pt=({float(pt[0]):.4f}, {float(pt[1]):.4f})")
        if not point_pass:
            all_pass = False
            # Only print unique failures
            for f in sorted(set(failures)):
                print(f)

    return all_pass


if __name__ == "__main__":
    print("=" * 70)
    print("  DIAGNOSTIC: C_eff symmetry tests")
    print("=" * 70)

    results = {}

    # Test minor symmetries (Test 1 from plan)
    results["minor_sym_raw"] = test_minor_symmetries(symmetrize=False)
    results["minor_sym_symmetrized"] = test_minor_symmetries(symmetrize=True)

    # Test reflection symmetry (Test 2 from plan)
    results["reflection_raw"] = test_reflection_symmetry(symmetrize=False)
    results["reflection_symmetrized"] = test_reflection_symmetry(symmetrize=True)

    # Summary
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
