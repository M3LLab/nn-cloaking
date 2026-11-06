"""Gradient checks for the pixel optimisation, on a coarse mesh.

    .venv/bin/python -m pytest tests/test_circular_adjoint.py -q
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from rayleigh_cloak.embedded.adjoint import adjoint_solve, imag_signs, material_sensitivity, residual_vjp
from rayleigh_cloak.embedded.config import load_embedded_config
from rayleigh_cloak.embedded.neural_field import make_neural_field
from rayleigh_cloak.embedded.optimize_pixels import TrainCase, loss_and_grad
from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("adj"))
    cfg = load_embedded_config("configs/circular_embedded/step1.yaml").with_updates(
        {"mesh.scale": 2.5, "cloak.radial_map": "quadratic", "cloak.eps": 0.05, "source.x": -3.5, "source.y": -2.0,
         "output_dir": out})
    grid = PolarPixelGrid.from_config(cfg, 12, 3)
    C, rho = sample_ideal(cfg, grid, "average")
    C = symmetrize_many(C)
    theta, field = make_neural_field(grid, C, rho, cfg.material.mu, hidden=16, layers=2, n_theta_modes=2,
                                     n_r_modes=1, output_scale=0.1)
    return cfg, grid, theta, field, C, rho, out


def _perturb(theta, seed):
    leaves, tree = jax.tree.flatten(theta)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(leaves))
    return jax.tree.unflatten(tree, [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)])


def _dot(a, b):
    return sum(float(np.sum(x * y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def test_network_starts_at_the_initial_material(setup):
    _, _, theta, field, C, rho, _ = setup
    Cd, rd = field.decode(theta)
    np.testing.assert_allclose(np.asarray(Cd), np.asarray(C), rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(np.asarray(rd), np.asarray(rho), rtol=1e-12)


def test_decoded_material_is_admissible(setup):
    _, _, theta, field, *_ = setup
    Cd, rd = field.decode(jax.tree.map(lambda a, b: a + 3.0 * b, theta, _perturb(theta, 3)))
    Cd = np.asarray(Cd)
    np.testing.assert_allclose(Cd, np.swapaxes(Cd, 1, 2), atol=1e-6)        # minor
    np.testing.assert_allclose(Cd, np.einsum("pijkl->pklij", Cd), atol=1e-6)  # major
    v = np.stack([Cd[:, 0, 0, 0, 0], Cd[:, 0, 0, 1, 1], Cd[:, 0, 0, 0, 1], Cd[:, 1, 1, 0, 0], Cd[:, 1, 1, 1, 1],
                  Cd[:, 1, 1, 0, 1], Cd[:, 0, 1, 0, 0], Cd[:, 0, 1, 1, 1], Cd[:, 0, 1, 0, 1]], axis=1).reshape(-1, 3, 3)
    assert np.linalg.eigvalsh(v).min() > 0
    assert np.asarray(rd).min() > 0


@pytest.mark.parametrize("loss", ["complex", "magnitude"])
def test_gradient_matches_finite_differences(setup, loss):
    cfg, grid, theta, field, _, _, out = setup
    case = TrainCase(cfg, grid, f"fd_{loss}", loss=loss, msh_dir=out)
    _, _, g, _ = loss_and_grad(theta, field, [case])
    for seed in (1, 2):
        v = _perturb(theta, seed)
        exact = _dot(g, v)
        h = 1e-4

        def J(sgn):
            th = jax.tree.map(lambda t, d: t + sgn * h * d, theta, v)
            C, r = field.decode(th)
            return case.loss_and_material_grad(C, r, want_grad=False)[0]

        fd = (J(+1) - J(-1)) / (2 * h)
        assert abs(exact - fd) <= 1e-5 * max(abs(fd), 1e-12), (exact, fd)
    case.backend.free()


def test_sensitivity_equals_jaxfem_reverse_mode(setup):
    cfg, grid, theta, field, C, rho, out = setup
    case = TrainCase(cfg, grid, "vjp", msh_dir=out)
    x, _ = case.solve(C, rho)
    rng = np.random.default_rng(0)
    u = x[:, 0].reshape(-1, 4)
    lam = rng.normal(size=u.shape)
    lam.ravel()[case.rows] = 0.0
    a, b = material_sensitivity(case.sd, u, lam)
    va, vb = residual_vjp(case.pp, u, lam)
    np.testing.assert_allclose(np.asarray(a), np.asarray(va), rtol=1e-8, atol=1e-8 * float(np.abs(va).max()))
    np.testing.assert_allclose(np.asarray(b), np.asarray(vb), rtol=1e-8, atol=1e-8 * float(np.abs(vb).max()))
    case.backend.free()


def test_adjoint_solves_the_transpose_system(setup):
    cfg, grid, theta, field, C, rho, out = setup
    case = TrainCase(cfg, grid, "adj", msh_dir=out)
    _, asm = case.solve(C, rho)
    n = asm.n
    g = np.random.default_rng(1).normal(size=n)
    lam = adjoint_solve(case.backend, g.copy(), asm.rows)
    free = np.ones(n, bool)
    free[asm.rows] = False
    r = (asm.A.T @ lam - g)[free]
    assert np.linalg.norm(r) < 1e-8 * np.linalg.norm(g[free])
    assert imag_signs(8).tolist() == [1, 1, -1, -1, 1, 1, -1, -1]
    case.backend.free()
