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
from rayleigh_cloak.embedded.optimize_pixels import BandCase, FrequencyCase, band_loss_and_grad, loss_and_grad
from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("adj"))
    cfg = load_embedded_config("configs/circular_embedded/step1.yaml").with_updates(
        {"mesh.scale": 2.5, "cloak.radial_map": "quadratic", "cloak.eps": 0.05, "output_dir": out})
    grid = PolarPixelGrid.from_config(cfg, 12, 3)
    C, rho = sample_ideal(cfg, grid, "average")
    C = symmetrize_many(C)
    theta, field = make_neural_field(grid, C, rho, cfg.material.mu, hidden=16, layers=2, n_theta_modes=2,
                                     n_r_modes=1, output_scale=0.1)
    return cfg, grid, theta, field, C, rho, out


SOURCES = {"train": [(-3.5, -2.0), (3.0, 1.0)], "val": [(0.0, 3.5)]}


def _case(cfg, grid, loss="complex"):
    return FrequencyCase(cfg, grid, SOURCES, loss=loss, msh_path=f"{cfg.output_dir}/_{loss}.msh")


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
    case = _case(cfg, grid, loss)
    _, _, g, _, _ = loss_and_grad(theta, field, [case])
    for seed in (1, 2):
        v = _perturb(theta, seed)
        exact = _dot(g, v)
        h = 1e-4

        def J(sgn):
            th = jax.tree.map(lambda t, d: t + sgn * h * d, theta, v)
            C, r = field.decode(th)
            return case.step(C, r, groups=())[0]

        fd = (J(+1) - J(-1)) / (2 * h)
        assert abs(exact - fd) <= 1e-5 * max(abs(fd), 1e-12), (exact, fd)
    case.backend.free()


def test_sensitivity_equals_jaxfem_reverse_mode(setup):
    cfg, grid, theta, field, C, rho, out = setup
    case = _case(cfg, grid)
    asm = case._factorize(C, rho)
    x = case._solve_group(asm, case.idx["train"])
    rng = np.random.default_rng(0)
    u = x[:, 0].reshape(-1, 4)
    lam = rng.normal(size=u.shape)
    lam.ravel()[case.rows] = 0.0
    a, b = material_sensitivity(case.sd, u, lam)
    va, vb = residual_vjp(case.pp, u, lam)
    np.testing.assert_allclose(np.asarray(a), np.asarray(va), rtol=1e-8, atol=1e-8 * float(np.abs(va).max()))
    np.testing.assert_allclose(np.asarray(b), np.asarray(vb), rtol=1e-8, atol=1e-8 * float(np.abs(vb).max()))
    case.backend.free()


def test_shared_mesh_force_matches_single_source_force(setup):
    cfg, grid, _, _, _, _, _ = setup
    case = _case(cfg, grid)
    p = case.pp.problem
    a = np.asarray(p.force_field_at((p._cfg.source.x, p._cfg.source.y), (0.0, 1.0)))
    b = np.asarray(p.force_field((0.0, 1.0)))
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())


def test_adjoint_solves_the_transpose_system(setup):
    cfg, grid, theta, field, C, rho, out = setup
    case = _case(cfg, grid)
    asm = case._factorize(C, rho)
    n = asm.n
    g = np.random.default_rng(1).normal(size=n)
    lam = adjoint_solve(case.backend, g.copy(), asm.rows)
    free = np.ones(n, bool)
    free[asm.rows] = False
    r = (case.backend.A.T @ lam - g)[free]      # the matrix actually factorised
    assert np.linalg.norm(r) < 1e-8 * np.linalg.norm(g[free])
    assert imag_signs(8).tolist() == [1, 1, -1, -1, 1, 1, -1, -1]
    case.backend.free()


def test_band_case_matches_a_separate_problem_and_its_gradient(setup):
    cfg, grid, theta, field, C, rho, out = setup
    hz = cfg.source.frequency_hz
    pool = [0.8, 1.0]                                  # f* with cfg at f* = 1; same mesh for both
    band = BandCase(cfg, grid, SOURCES, pool, msh_path=f"{out}/_band.msh")
    # a separate problem built directly at f* = 0.8, on the same mesh (h fixed by rescaling n per wavelength)
    c08 = cfg.with_updates({"source.frequency_hz": 0.8 * hz, "mesh.elems_per_wavelength": cfg.mesh.elems_per_wavelength / 0.8})
    sep = FrequencyCase(c08, grid, SOURCES, msh_path=f"{out}/_sep.msh")
    assert band.n_cells == sep.n_cells
    band.select(0.8)
    jb = band.step(C, rho, groups=())[0]
    js = sep.step(C, rho, groups=())[0]
    np.testing.assert_allclose(jb, js, rtol=1e-8)
    # gradient across the pool against finite differences
    _, _, g, _ = band_loss_and_grad(theta, field, band, pool)
    v = _perturb(theta, 7)
    h = 1e-4

    def J(sgn):
        th = jax.tree.map(lambda t, d: t + sgn * h * d, theta, v)
        return band_loss_and_grad(th, field, band, pool)[0]

    fd = (J(+1) - J(-1)) / (2 * h)
    assert abs(_dot(g, v) - fd) <= 1e-5 * abs(fd), (_dot(g, v), fd)
