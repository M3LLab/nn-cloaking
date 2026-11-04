"""Audit of the regularised circular-cloak transformation and BGM material.

Checks, for both radial maps and several eps:

1. the radial map hits its end points and inverts correctly;
2. the closed-form Cartesian F equals the autodiff Jacobian of the forward
   point map X -> x, evaluated at X = Xi^{-1}(x);
3. det F = F_r F_t and rho' = rho / det F;
4. the Cartesian push-forward C' = F F C / J equals the closed-form polar
   tensor rotated (full fourth-order rotation) into the Cartesian frame;
5. C' has major but not minor symmetry (Cosserat, not Cauchy);
6. strain-energy invariance: C'(x) : G_x : G_x * J == C : G_X : G_X for any
   virtual displacement gradient G_X, with G_x = G_X F^{-1};
7. eps = 0 + linear reproduces the legacy singular-map formula.

Run:  .venv/bin/python -m pytest tests/test_circular_transform.py -q
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from rayleigh_cloak.geometry.circular import CircularCloakGeometry, RadialMap
from rayleigh_cloak.materials import (
    C_eff, C_iso, C_pushforward, C_radial_polar, rho_eff, rotate_C,
)

LAM, MU, RHO = 144e6, 144e6, 1600.0
A, B = 0.5, 1.5
XC, YC = 0.3, -0.2
CASES = [(kind, eps) for kind in ("linear", "quadratic")
         for eps in (0.0, 0.005, 0.05, 0.2)]


def _points(geo, n_r=6, n_t=7, seed=0):
    rng = np.random.default_rng(seed)
    rs = np.linspace(geo.a * 1.0005, geo.b * 0.9995, n_r)
    ts = rng.uniform(0, 2 * np.pi, n_t)
    return [jnp.array([geo.x_c + r * np.cos(t), geo.y_c + r * np.sin(t)])
            for r in rs for t in ts]


@pytest.mark.parametrize("kind,eps", CASES)
def test_radial_map_endpoints_and_inverse(kind, eps):
    m = RadialMap(a=A, b=B, eps=eps, kind=kind)
    assert np.isclose(float(m.r_of_R(eps)), A, rtol=0, atol=1e-13)
    assert np.isclose(float(m.r_of_R(B)), B, rtol=0, atol=1e-13)
    R = jnp.linspace(eps, B, 101)
    assert np.all(np.diff(np.asarray(m.r_of_R(R))) > 0), "map not monotone"
    np.testing.assert_allclose(np.asarray(m.R_of_r(m.r_of_R(R))), np.asarray(R),
                               atol=1e-12)
    dr = jax.vmap(jax.grad(lambda s: m.r_of_R(s)))(R)
    np.testing.assert_allclose(np.asarray(dr), np.asarray(m.drdR(R)), rtol=1e-12)
    if kind == "quadratic":
        assert np.isclose(float(m.drdR(B)), 1.0, atol=1e-13)


@pytest.mark.parametrize("kind,eps", CASES)
def test_F_matches_autodiff_jacobian(kind, eps):
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=eps, radial_map=kind)
    jac = jax.jacfwd(geo.physical_point)
    for x in _points(geo):
        X = geo.virtual_point(x)
        np.testing.assert_allclose(np.asarray(geo.physical_point(X)), np.asarray(x),
                                   atol=1e-12)
        F_ad = jac(X)
        F_cf = geo.F_tensor(x)
        np.testing.assert_allclose(np.asarray(F_cf), np.asarray(F_ad), rtol=1e-9,
                                   atol=1e-12)


@pytest.mark.parametrize("kind,eps", CASES)
def test_det_and_density(kind, eps):
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=eps, radial_map=kind)
    for x in _points(geo):
        _, _, F_r, F_t, _ = geo.polar_frame(x)
        J = float(jnp.linalg.det(geo.F_tensor(x)))
        np.testing.assert_allclose(J, float(F_r * F_t), rtol=1e-12)
        np.testing.assert_allclose(float(rho_eff(x, geo, RHO)), RHO / J, rtol=1e-12)


@pytest.mark.parametrize("kind,eps", CASES)
def test_cartesian_equals_rotated_polar(kind, eps):
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=eps, radial_map=kind)
    C0 = C_iso(LAM, MU)
    for x in _points(geo):
        _, _, F_r, F_t, Q = geo.polar_frame(x)
        C_cart = C_eff(x, geo, C0)
        C_pol = rotate_C(C_radial_polar(LAM, MU, F_r, F_t), Q)
        np.testing.assert_allclose(np.asarray(C_cart), np.asarray(C_pol),
                                   rtol=1e-10, atol=1e-6 * LAM)
        # back-rotation recovers the polar closed form
        C_back = rotate_C(C_cart, Q.T)
        np.testing.assert_allclose(np.asarray(C_back),
                                   np.asarray(C_radial_polar(LAM, MU, F_r, F_t)),
                                   rtol=1e-10, atol=1e-6 * LAM)


@pytest.mark.parametrize("kind,eps", CASES)
def test_major_not_minor_symmetry(kind, eps):
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=eps, radial_map=kind)
    C0 = C_iso(LAM, MU)
    worst_minor = 0.0
    for x in _points(geo):
        C = np.asarray(C_eff(x, geo, C0))
        np.testing.assert_allclose(C, np.einsum("ijkl->klij", C), atol=1e-6 * LAM)
        worst_minor = max(worst_minor,
                          np.abs(C - np.einsum("ijkl->jikl", C)).max() / LAM)
    assert worst_minor > 1e-3, "push-forward unexpectedly minor-symmetric"


@pytest.mark.parametrize("kind,eps", CASES)
def test_strain_energy_invariance(kind, eps):
    """W_phys * J == W_virtual for arbitrary (non-symmetric) displacement gradients."""
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=eps, radial_map=kind)
    C0 = C_iso(LAM, MU)
    rng = np.random.default_rng(1)
    for x in _points(geo):
        F = geo.F_tensor(x)
        J = jnp.linalg.det(F)
        Cp = C_pushforward(F, C0)
        for _ in range(3):
            G_X = jnp.array(rng.normal(size=(2, 2)))   # G_X[i, J] = dU_i/dX_J
            G_x = G_X @ jnp.linalg.inv(F)              # du_i/dx_j = dU_i/dX_J dX_J/dx_j
            W_v = jnp.einsum("ijkl,ij,kl->", C0, G_X, G_X)
            W_p = jnp.einsum("ijkl,ij,kl->", Cp, G_x, G_x) * J
            # eps = 0 puts F_t ~ 1e3 next to r = a, so allow round-off there
            np.testing.assert_allclose(float(W_p), float(W_v), rtol=1e-8)


def test_eps0_linear_matches_legacy_formula():
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC)
    for x in _points(geo):
        dx = np.asarray(x) - np.array([XC, YC])
        r = np.linalg.norm(dx)
        xh = dx / r
        R = B * (r - A) / (B - A)
        F_legacy = (r * np.eye(2) - A * np.outer(xh, xh)) / R
        np.testing.assert_allclose(np.asarray(geo.F_tensor(x)), F_legacy, rtol=1e-12)


def test_identity_outside_annulus():
    geo = CircularCloakGeometry(ri=A, rc=B, x_c=XC, y_c=YC, eps=0.01)
    C0 = C_iso(LAM, MU)
    for r in (0.2, 1.6, 3.0):
        x = jnp.array([XC + r, YC])
        np.testing.assert_allclose(np.asarray(geo.F_tensor(x)), np.eye(2))
        np.testing.assert_allclose(np.asarray(C_eff(x, geo, C0)), np.asarray(C0))
