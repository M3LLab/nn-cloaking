"""Neural field mapping pixel position to an admissible Cauchy material.

Each pixel gets a minor- and major-symmetric, positive-definite stiffness and a positive density,
by construction: M = L L^T for the 3 x 3 Voigt matrix [C11 C12 C16; C12 C22 C26; C16 C26 C66],
and rho = rho_0 exp(.). The last layer starts at zero, so the initial design is exactly the initial
pixel material; the network learns a correction, scaled by ``output_scale``.

MLP and Adam come from the existing pipeline (``neural_reparam``, ``optimize``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.embedded.pixels import PolarPixelGrid
from rayleigh_cloak.materials import C_to_flat6cauchy, flat6cauchy_to_C
from rayleigh_cloak.neural_reparam import init_mlp, mlp_forward

_TRIL = (np.array([0, 1, 1, 2, 2, 2]), np.array([0, 0, 1, 0, 1, 2]))


def _voigt_from_C(C):
    f = C_to_flat6cauchy(C)   # C11, C22, C66, C12, C16, C26
    return jnp.array([[f[0], f[3], f[4]], [f[3], f[1], f[5]], [f[4], f[5], f[2]]])


def _C_from_voigt(M):
    return flat6cauchy_to_C(jnp.array([M[0, 0], M[1, 1], M[2, 2], M[0, 1], M[0, 2], M[1, 2]]))


def polar_features(grid: PolarPixelGrid, n_theta_modes: int, n_r_modes: int) -> jnp.ndarray:
    """Periodic in theta, Fourier in normalised radius, plus their products."""
    r, t = grid.centres_polar()
    s = (r - grid.a) / (grid.b - grid.a)
    th = [f(m * t) for m in range(1, n_theta_modes + 1) for f in (np.cos, np.sin)]
    rd = [s] + [f(np.pi * k * s) for k in range(1, n_r_modes + 1) for f in (np.cos, np.sin)]
    feats = rd + th + [a * b for a in rd[:3] for b in th]
    return jnp.asarray(np.stack(feats, axis=1))


@dataclass
class NeuralField:
    features: jnp.ndarray
    L0: jnp.ndarray          # (n_pixels, 3, 3), Cholesky factor of the scaled initial Voigt matrix
    rho0: jnp.ndarray
    scale: float             # stiffness scale (mu)
    output_scale: float

    def decode(self, theta):
        """Return (C_pix, rho_pix) for network parameters ``theta``."""
        out = self.output_scale * mlp_forward(theta, self.features)     # (n_pixels, 7)
        d0 = jnp.diagonal(self.L0, axis1=1, axis2=2)
        off = jnp.zeros_like(self.L0).at[:, _TRIL[0], _TRIL[1]].set(out[:, :6])
        rel = jnp.einsum("pi,pj->pij", d0, d0) ** 0.5
        L = self.L0 + jnp.tril(off, -1) * rel
        diag = d0 * jnp.exp(jnp.diagonal(off, axis1=1, axis2=2))
        L = L.at[:, jnp.arange(3), jnp.arange(3)].set(diag)
        M = self.scale * L @ jnp.swapaxes(L, 1, 2)
        return jax.vmap(_C_from_voigt)(M), self.rho0 * jnp.exp(out[:, 6])


def make_neural_field(grid, C_init, rho_init, scale, hidden=128, layers=3, n_theta_modes=8,
                      n_r_modes=4, seed=0, output_scale=0.1):
    """Build the field and network starting exactly at (C_init, rho_init), which must be Cauchy."""
    feats = polar_features(grid, n_theta_modes, n_r_modes)
    M0 = jax.vmap(_voigt_from_C)(jnp.asarray(C_init)) / scale
    L0 = jnp.linalg.cholesky(M0)
    if not bool(jnp.all(jnp.isfinite(L0))):
        raise ValueError("initial pixel stiffness is not positive definite")
    theta = init_mlp(jax.random.PRNGKey(seed), [feats.shape[1]] + [hidden] * layers + [7])
    theta[-1] = {"W": jnp.zeros_like(theta[-1]["W"]), "b": jnp.zeros_like(theta[-1]["b"])}
    return theta, NeuralField(feats, L0, jnp.asarray(rho_init), float(scale), float(output_scale))
