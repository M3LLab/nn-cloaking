"""Differentiable shape parameterisation for open-geometry cloak optimisation.

Decouples "where is the cloak?" from "what material is in it?".  The cloak
footprint is represented by a per-cell logit ``s``; the occupancy weight
``m = sigmoid(beta * s) in [0, 1]`` blends the trainable cell material toward
a fixed background:

    C_flat_eff = m * cell_C_flat + (1 - m) * C0_flat
    rho_eff    = m * cell_rho    + (1 - m) * rho0

At ``m = 1`` the cell behaves like an ordinary cell-based cloak cell; at
``m = 0`` it is indistinguishable from background.  Gradients flow through
both the material values and the logits, so shape and material co-optimise.

The module intentionally knows nothing about FEM: it operates on the
per-cell arrays that the existing :class:`RayleighCloakProblem` already
consumes.  That keeps geometry parameterisation separate from physics code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def init_logits_from_cloak_mask(
    cloak_mask: np.ndarray,
    magnitude: float = 3.0,
) -> jnp.ndarray:
    """Seed per-cell logits so the initial sigmoid matches the boolean mask.

    ``magnitude`` controls how decisive the initial occupancy is; avoid
    values so large that ``sigmoid(beta * magnitude)`` saturates (derivative
    vanishes).  Default ``3.0`` gives ``sigmoid(3) ≈ 0.95`` with ``beta=1``,
    preserving a usable gradient on day zero.
    """
    mask = np.asarray(cloak_mask, dtype=bool)
    logits = np.where(mask, magnitude, -magnitude).astype(np.float32)
    return jnp.asarray(logits)


def occupancy(logits: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Per-cell occupancy ``m = sigmoid(beta * logits)`` in [0, 1]."""
    return jax.nn.sigmoid(beta * logits)


def apply_shape_mask(
    cell_C_flat: jnp.ndarray,   # (n_cells, n_C_params)
    cell_rho: jnp.ndarray,      # (n_cells,)
    logits: jnp.ndarray,        # (n_cells,)
    C0_flat: jnp.ndarray,       # (n_C_params,)
    rho0: float,
    beta: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Blend trainable cell materials toward the background under the mask.

    Returns a ``(cell_C_flat_eff, cell_rho_eff)`` pair with the same shapes
    as the inputs, ready to feed into the existing FEM forward.
    """
    m = occupancy(logits, beta)
    m_mat = m[:, None]
    C_eff = m_mat * cell_C_flat + (1.0 - m_mat) * C0_flat[None, :]
    rho_eff = m * cell_rho + (1.0 - m) * rho0
    return C_eff, rho_eff


def mask_smoothness(
    logits: jnp.ndarray,
    neighbor_pairs: jnp.ndarray,
) -> jnp.ndarray:
    """Squared difference of logits across 4-connected neighbour pairs.

    A cheap perimeter-like penalty: discourages speckled masks without
    committing to a binarisation schedule.  Scale with ``lambda_mask_smooth``.
    """
    if neighbor_pairs.shape[0] == 0:
        return jnp.float32(0.0)
    s_i = logits[neighbor_pairs[:, 0]]
    s_j = logits[neighbor_pairs[:, 1]]
    return jnp.mean((s_i - s_j) ** 2)


def all_neighbor_pairs(n_x: int, n_y: int) -> np.ndarray:
    """4-connected neighbour pairs over the full ``n_x × n_y`` grid.

    The ``CellDecomposition.get_neighbor_pairs`` helper only covers pairs
    whose cells are both in the initial cloak mask — useful for material
    smoothness but unsuitable for the shape logits, which must be free to
    reshape across the original boundary.  Flat index: ``ix * n_y + iy``.
    """
    pairs = []
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            if ix + 1 < n_x:
                pairs.append((idx, (ix + 1) * n_y + iy))
            if iy + 1 < n_y:
                pairs.append((idx, ix * n_y + (iy + 1)))
    if not pairs:
        return np.empty((0, 2), dtype=int)
    return np.asarray(pairs, dtype=int)
