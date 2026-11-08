"""The incremental assembly equals a full jax-fem assembly, and training steps agree with the slow path.

    .venv/bin/python -m pytest tests/test_circular_fast_assembly.py -q
"""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from rayleigh_cloak.embedded.adjoint import assemble
from rayleigh_cloak.embedded.config import load_embedded_config
from rayleigh_cloak.embedded.optimize_pixels import BandCase, FrequencyCase
from rayleigh_cloak.embedded.pixels import PolarPixelGrid, sample_ideal, symmetrize_many

SOURCES = {"train": [(-3.5, -2.0), (3.0, 1.0)], "val": [(0.0, 3.5)]}


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("fast"))
    cfg = load_embedded_config("configs/circular_embedded/step1.yaml").with_updates(
        {"mesh.scale": 2.5, "cloak.radial_map": "quadratic", "cloak.eps": 0.05, "output_dir": out})
    grid = PolarPixelGrid.from_config(cfg, 12, 3)
    C, rho = sample_ideal(cfg, grid, "average")
    return cfg, grid, symmetrize_many(C), rho, out


def test_incremental_matrix_equals_full_assembly(setup):
    cfg, grid, C, rho, out = setup
    case = BandCase(cfg, grid, SOURCES, [0.8, 1.0], msh_path=f"{out}/_a.msh")
    for f in (1.0, 0.8):
        case.select(f)
        case._factorize(C, rho)
        case.pp.set_pixels(C, case._rho_scale * rho)
        A_full = assemble(case.pp.problem).A
        A_fast = case.backend.A
        diff = (A_fast - A_full)
        assert abs(diff).max() <= 1e-10 * abs(A_full).max(), f
    case.backend.free()


def test_fast_and_slow_steps_agree(setup):
    cfg, grid, C, rho, out = setup
    fast = FrequencyCase(cfg, grid, SOURCES, msh_path=f"{out}/_b.msh", fast=True)
    slow = FrequencyCase(cfg, grid, SOURCES, msh_path=f"{out}/_c.msh", fast=False)
    jf, hf, gCf, grf, mf = fast.step(C, rho, groups=("val",))
    js, hs, gCs, grs, ms = slow.step(C, rho, groups=("val",))
    np.testing.assert_allclose(jf, js, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(gCf), np.asarray(gCs), rtol=1e-7, atol=1e-9 * float(np.abs(gCs).max()))
    np.testing.assert_allclose(np.asarray(grf), np.asarray(grs), rtol=1e-7, atol=1e-9 * float(np.abs(grs).max()))
    np.testing.assert_allclose(mf["val"], ms["val"], rtol=1e-9)
    assert set(fast.timing) >= {"assemble", "factorize", "solve", "adjoint", "sensitivity"}
