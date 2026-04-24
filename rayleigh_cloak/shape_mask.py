"""Differentiable shape parameterisation for open-geometry cloak optimisation.

Decouples "where is the cloak?" from "what material is in it?".  The cloak
footprint is represented by a per-cell logit ``s``; the occupancy weight
``m = sigmoid(beta * s) in [0, 1]`` blends the trainable cell material toward
a fixed background:

    C_flat_eff = m**p * cell_C_flat + (1 - m**p) * C0_flat
    rho_eff    = m**p * cell_rho    + (1 - m**p) * rho0

At ``m = 1`` the cell behaves like an ordinary cell-based cloak cell; at
``m = 0`` it is indistinguishable from background.  The SIMP exponent ``p``
(default ``1``) penalises intermediate values — raising it pushes grey
regions toward pure cloak or pure background during training.

Gradients flow through both the material values and the logits, so shape and
material co-optimise.

The module intentionally knows nothing about FEM: it operates on the
per-cell arrays that the existing :class:`RayleighCloakProblem` already
consumes.  That keeps geometry parameterisation separate from physics code.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── logit init ───────────────────────────────────────────────────────


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


# ── spatial smoothing ────────────────────────────────────────────────


def _gaussian_kernel_1d(sigma: float, radius: int) -> jnp.ndarray:
    xs = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-0.5 * (xs / sigma) ** 2)
    return k / k.sum()


def smooth_logits(
    logits: jnp.ndarray,   # (n_x * n_y,)
    n_x: int,
    n_y: int,
    sigma: float,
    radius: int | None = None,
) -> jnp.ndarray:
    """Separable Gaussian filter on the logits grid.

    Cheap stand-in for the PDE-based Helmholtz filter used in topology
    optimisation.  Suppresses thin features and single-cell islands, which
    usually correspond to disconnected components after binarisation.

    ``sigma`` is in cell units; ``radius <= 0`` or ``sigma <= 0`` disables
    the filter (pass-through).  Differentiable.
    """
    if sigma <= 0.0:
        return logits
    if radius is None:
        radius = int(3 * sigma + 0.5)
    # Clamp so the kernel never exceeds the signal in either direction —
    # otherwise jnp.convolve(mode="same") returns max(M, N), mangling shape.
    max_r = (min(n_x, n_y) - 1) // 2
    radius = min(radius, max_r)
    if radius <= 0:
        return logits
    k = _gaussian_kernel_1d(sigma, radius)
    g = logits.reshape(n_x, n_y)
    g = jax.vmap(lambda row: jnp.convolve(row, k, mode="same"))(g)         # along y
    g = jax.vmap(lambda col: jnp.convolve(col, k, mode="same"))(g.T).T      # along x
    return g.reshape(-1)


# ── occupancy + blend ────────────────────────────────────────────────


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
    simp_p: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Blend trainable cell materials toward the background under the mask.

    The SIMP exponent ``simp_p`` is applied to the occupancy before blending.
    ``simp_p = 1`` recovers a plain convex combination (default).  Values
    ``> 1`` (typically ``3``) make intermediate ``m`` contribute less
    stiffness per unit "material cost", which in concert with a binarisation
    regulariser nudges ``m`` toward pure 0 or 1.
    """
    m = occupancy(logits, beta)
    m_p = m ** simp_p if simp_p != 1.0 else m
    m_mat = m_p[:, None]
    C_eff = m_mat * cell_C_flat + (1.0 - m_mat) * C0_flat[None, :]
    rho_eff = m_p * cell_rho + (1.0 - m_p) * rho0
    return C_eff, rho_eff


# ── regularisers ─────────────────────────────────────────────────────


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


def binarization_penalty(m: jnp.ndarray) -> jnp.ndarray:
    """``mean(m * (1 - m))`` — zero at ``m ∈ {0, 1}``, max at ``m = 0.5``.

    Pulls intermediate occupancy toward the endpoints.  Gradients are small
    near the endpoints by construction — use together with a β ramp and/or
    SIMP exponent for effective binarisation.
    """
    return jnp.mean(m * (1.0 - m))


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


# ── post-hoc topology cleanup (numpy-only; not differentiable) ───────


def largest_connected_component(
    mask_2d: np.ndarray,
    connectivity: int = 1,
) -> np.ndarray:
    """Return a boolean array with only the largest 4-connected component.

    ``mask_2d`` may be bool or float; values ``> 0.5`` are treated as solid.
    ``connectivity = 1`` → 4-connected, ``2`` → 8-connected.  Returns an
    array with the same shape; all other components are zeroed out.  If the
    input has no solid cells the return is all-``False``.
    """
    from scipy.ndimage import label

    solid = np.asarray(mask_2d) > 0.5
    if not solid.any():
        return np.zeros_like(solid)

    if connectivity == 1:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    elif connectivity == 2:
        structure = np.ones((3, 3), dtype=bool)
    else:
        raise ValueError("connectivity must be 1 or 2")

    labels, n_labels = label(solid, structure=structure)
    if n_labels == 0:
        return np.zeros_like(solid)

    # Count cells per label (labels start at 1; 0 is background)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # ignore background
    biggest = int(counts.argmax())
    return labels == biggest
