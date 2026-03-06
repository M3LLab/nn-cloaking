"""Absorbing-layer (Rayleigh-damping) profile.

Implements a position-dependent damping ratio xi(x) that ramps from 0 at the
physical/PML interface to xi_max at the outer boundary (quadratic by default).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from rayleigh_cloak.config import DerivedParams


def make_xi_profile(params: DerivedParams) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a closure ``xi(x)`` capturing domain parameters.

    The returned function is JAX-traceable and suitable for ``jax.vmap``.
    """
    x_off = params.x_off
    y_off = params.y_off
    W = params.W
    L_pml = params.L_pml
    xi_max = params.xi_max
    pml_pow = params.pml_pow

    def xi_profile(x: jnp.ndarray) -> jnp.ndarray:
        # lateral attenuation
        d_left = jnp.maximum(x_off - x[0], 0.0)
        d_right = jnp.maximum(x[0] - (x_off + W), 0.0)
        xi_x = xi_max * (jnp.maximum(d_left, d_right) / L_pml) ** pml_pow

        # vertical attenuation (bottom only)
        d_bot = jnp.maximum(y_off - x[1], 0.0)
        xi_y = xi_max * (d_bot / L_pml) ** pml_pow

        return xi_x + xi_y

    return xi_profile
